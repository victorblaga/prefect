"""Tests for cancellation cleanup services."""

import logging
from collections.abc import Awaitable
from datetime import datetime, timedelta
from typing import Any, Callable
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

import pytest
from docket import Docket, Worker
from sqlalchemy.ext.asyncio import AsyncSession

from prefect.client.schemas.worker_channel import CANCELLING_TIMEOUT_TEARDOWN
from prefect.server import models, schemas
from prefect.server.database import orm_models, provide_database_interface
from prefect.server.orchestration.core_policy import CoreFlowPolicy
from prefect.server.schemas import responses, states
from prefect.server.schemas.core import Deployment, Flow, FlowRun
from prefect.server.services.cancellation_cleanup import (
    CANCELLED_BEFORE_START_MESSAGE,
    CANCELLING_TIMEOUT_CANCELLED_MESSAGE,
    cancel_before_start_key,
    cancel_child_task_runs,
    cancel_flow_run_before_start,
    cancel_subflow_run,
    cancelling_timeout_check_key,
    ensure_cancelling_timeout_checks,
    handle_cancelling_timeout,
    maybe_schedule_cancelling_timeout_check_for_state,
    monitor_cancelled_flow_runs,
    monitor_subflow_runs,
    schedule_cancelling_timeout_check,
)
from prefect.server.worker_communication.cleanup_queue import CleanupQueueMessage
from prefect.server.worker_communication.cleanup_queue.memory import WorkerCleanupQueue
from prefect.settings import (
    PREFECT_SERVER_SERVICES_CANCELLATION_CLEANUP_CANCELLING_TIMEOUT_SECONDS,
    PREFECT_SERVER_SERVICES_CANCELLATION_CLEANUP_ENABLED,
    temporary_settings,
)
from prefect.types._datetime import Duration, now

pytestmark = pytest.mark.clear_db

NON_TERMINAL_STATE_CONSTRUCTORS: dict[states.StateType, Any] = {
    states.StateType.SCHEDULED: states.Scheduled,
    states.StateType.PENDING: states.Pending,
    states.StateType.RUNNING: states.Running,
    states.StateType.PAUSED: states.Paused,
    states.StateType.CANCELLING: states.Cancelling,
}

TERMINAL_STATE_CONSTRUCTORS: dict[states.StateType, Any] = {
    states.StateType.COMPLETED: states.Completed,
    states.StateType.FAILED: states.Failed,
    states.StateType.CRASHED: states.Crashed,
    states.StateType.CANCELLED: states.Cancelled,
}

THE_PAST = now("UTC") - Duration(hours=5)
THE_ANCIENT_PAST = now("UTC") - Duration(days=100)


@pytest.fixture
async def cancelled_flow_run(session: AsyncSession, flow: Flow):
    async with session.begin():
        return await models.flow_runs.create_flow_run(
            session=session,
            flow_run=schemas.core.FlowRun(
                flow_id=flow.id, state=states.Cancelled(), end_time=THE_PAST
            ),
        )


@pytest.fixture
async def old_cancelled_flow_run(session: AsyncSession, flow: Flow):
    async with session.begin():
        return await models.flow_runs.create_flow_run(
            session=session,
            flow_run=schemas.core.FlowRun(
                flow_id=flow.id, state=states.Cancelled(), end_time=THE_ANCIENT_PAST
            ),
        )


@pytest.fixture
def cleanup_queue() -> WorkerCleanupQueue:
    # The in-memory queue is a per-process singleton, so tasks run by a real
    # Docket worker in these tests enqueue into this same instance.
    queue = WorkerCleanupQueue()
    queue.clear()
    return queue


@pytest.fixture
async def cancelling_flow_run_maker(
    session: AsyncSession, flow: Flow, work_pool: Any
) -> Callable[..., Awaitable[Any]]:
    async def make_flow_run(
        *,
        state: states.State | None = None,
        infrastructure_pid: str | None = "infra-pid-1",
        start_time: datetime | None = None,
        use_default_work_queue: bool = True,
        work_pool_override: Any | None = None,
    ) -> Any:
        selected_work_pool = work_pool_override or work_pool
        async with session.begin():
            return await models.flow_runs.create_flow_run(
                session=session,
                flow_run=schemas.core.FlowRun(
                    flow_id=flow.id,
                    state=state
                    or states.Cancelling(timestamp=now("UTC") - Duration(seconds=120)),
                    infrastructure_pid=infrastructure_pid,
                    start_time=start_time,
                    work_queue_id=(
                        selected_work_pool.default_queue_id
                        if use_default_work_queue
                        else None
                    ),
                ),
            )

    return make_flow_run


@pytest.fixture
async def orphaned_task_run_maker(session: AsyncSession):
    async def task_run_maker(flow_run: FlowRun, state_constructor: Any):
        async with session.begin():
            return await models.task_runs.create_task_run(
                session=session,
                task_run=schemas.core.TaskRun(
                    flow_run_id=flow_run.id,
                    task_key="a task",
                    dynamic_key="a dynamic key",
                    state=state_constructor(),
                ),
            )

    return task_run_maker


@pytest.fixture
async def orphaned_subflow_run_maker(session: AsyncSession, flow: Flow):
    async def subflow_run_maker(flow_run: FlowRun, state_constructor: Any):
        async with session.begin():
            virtual_task = await models.task_runs.create_task_run(
                session=session,
                task_run=schemas.core.TaskRun(
                    flow_run_id=flow_run.id,
                    task_key="a virtual task",
                    dynamic_key="a virtual dynamic key",
                    state=state_constructor(),
                ),
            )

            return await models.flow_runs.create_flow_run(
                session=session,
                flow_run=schemas.core.FlowRun(
                    flow_id=flow.id,
                    parent_task_run_id=virtual_task.id,
                    state=state_constructor(),
                    end_time=THE_PAST,
                ),
            )

    return subflow_run_maker


@pytest.fixture
async def orphaned_subflow_run_from_deployment_maker(
    session: AsyncSession, flow: Flow, deployment: Deployment
):
    async def subflow_run_maker(flow_run: FlowRun, state_constructor: Any):
        async with session.begin():
            virtual_task = await models.task_runs.create_task_run(
                session=session,
                task_run=schemas.core.TaskRun(
                    flow_run_id=flow_run.id,
                    task_key="a virtual task for subflow from deployment",
                    dynamic_key="a virtual dynamic key for subflow from deployment",
                    state=state_constructor(),
                ),
            )

            return await models.flow_runs.create_flow_run(
                session=session,
                flow_run=schemas.core.FlowRun(
                    flow_id=flow.id,
                    parent_task_run_id=virtual_task.id,
                    state=state_constructor(),
                    end_time=THE_PAST,
                    deployment_id=deployment.id,
                    infrastructure_pid="my-pid-42",
                ),
            )

    return subflow_run_maker


async def test_all_state_types_are_tested():
    assert set(NON_TERMINAL_STATE_CONSTRUCTORS.keys()).union(
        set(TERMINAL_STATE_CONSTRUCTORS.keys())
    ) == set(states.StateType)


class RetryRecorder:
    def __init__(self) -> None:
        self.when: datetime | None = None

    def at(self, when: datetime) -> None:
        self.when = when


async def _handle_cancelling_timeout_for_run(
    flow_run: Any,
    cleanup_queue: WorkerCleanupQueue,
    *,
    flow_run_state_id: Any | None = None,
    timeout_cancelled_state_id: Any | None = None,
    retry: RetryRecorder | None = None,
):
    current_state_id = flow_run_state_id or flow_run.state_id
    assert current_state_id is not None
    timeout_state_id = timeout_cancelled_state_id or uuid4()

    message = await handle_cancelling_timeout(
        flow_run_id=flow_run.id,
        flow_run_state_id=current_state_id,
        timeout_cancelled_state_id=timeout_state_id,
        db=provide_database_interface(),
        cleanup_queue=cleanup_queue,
        retry=retry or RetryRecorder(),
    )
    return message, timeout_state_id


async def _cancel_flow_run_before_start(
    flow_run: orm_models.FlowRun,
    cleanup_queue: WorkerCleanupQueue,
    *,
    flow_run_state_id: UUID | None = None,
) -> CleanupQueueMessage | None:
    current_state_id = flow_run_state_id or flow_run.state_id
    assert current_state_id is not None

    return await cancel_flow_run_before_start(
        flow_run_id=flow_run.id,
        flow_run_state_id=current_state_id,
        db=provide_database_interface(),
        cleanup_queue=cleanup_queue,
    )


async def _run_background_worker(docket_name: str) -> None:
    """Run the tasks due on a Docket the way the server's background worker does."""
    # The server's background worker knows only the registered task functions.
    async with Docket(name=docket_name, url="memory://") as worker_docket:
        worker_docket.register_collection(
            "prefect.server.api.background_workers:task_functions"
        )
        async with Worker(
            worker_docket,
            minimum_check_interval=timedelta(milliseconds=10),
            scheduling_resolution=timedelta(milliseconds=10),
        ) as worker:
            await worker.run_until_finished()


async def test_schedule_cancelling_timeout_check_replaces_per_run_task():
    flow_run_id = uuid4()
    first_state_id = uuid4()
    second_state_id = uuid4()
    first_deadline = now("UTC") + Duration(minutes=5)
    second_deadline = now("UTC") + Duration(minutes=10)
    async with Docket(name=f"test-{uuid4()}", url="memory://") as docket:
        with temporary_settings(
            {
                PREFECT_SERVER_SERVICES_CANCELLATION_CLEANUP_ENABLED: True,
                PREFECT_SERVER_SERVICES_CANCELLATION_CLEANUP_CANCELLING_TIMEOUT_SECONDS: 60,
            }
        ):
            await maybe_schedule_cancelling_timeout_check_for_state(
                docket=docket,
                flow_run_id=flow_run_id,
                state=states.Cancelling(
                    id=first_state_id,
                    timestamp=first_deadline - Duration(seconds=60),
                ),
            )
        snapshot = await docket.snapshot()
        task_keys = [task.key for task in snapshot.future]
        task_keys.extend(task.key for task in snapshot.running)
        assert task_keys.count(cancelling_timeout_check_key(flow_run_id)) == 1

        await schedule_cancelling_timeout_check(
            docket=docket,
            flow_run_id=flow_run_id,
            flow_run_state_id=second_state_id,
            when=second_deadline,
        )
        snapshot = await docket.snapshot()

    task_keys = [task.key for task in snapshot.future]
    task_keys.extend(task.key for task in snapshot.running)
    assert task_keys.count(cancelling_timeout_check_key(flow_run_id)) == 1


async def test_cancelling_state_schedules_cancellation_before_start_and_timeout_check():
    flow_run_id = uuid4()
    state = states.Cancelling(id=uuid4(), timestamp=now("UTC") - Duration(seconds=5))
    assert state.timestamp is not None
    async with Docket(name=f"test-{uuid4()}", url="memory://") as docket:
        with temporary_settings(
            {
                PREFECT_SERVER_SERVICES_CANCELLATION_CLEANUP_ENABLED: True,
                PREFECT_SERVER_SERVICES_CANCELLATION_CLEANUP_CANCELLING_TIMEOUT_SECONDS: 60,
            }
        ):
            await maybe_schedule_cancelling_timeout_check_for_state(
                docket=docket, flow_run_id=flow_run_id, state=state
            )
        snapshot = await docket.snapshot()

    scheduled = {task.key: task for task in snapshot.future}
    cancel_before_start = scheduled[cancel_before_start_key(flow_run_id)]
    timeout_check = scheduled[cancelling_timeout_check_key(flow_run_id)]
    assert cancel_before_start.when == state.timestamp
    assert cancel_before_start.kwargs == {
        "flow_run_id": flow_run_id,
        "flow_run_state_id": state.id,
    }
    assert timeout_check.when == state.timestamp + Duration(seconds=60)


async def test_cancelling_run_that_never_started_is_cancelled_and_torn_down(
    cleanup_queue: WorkerCleanupQueue,
    cancelling_flow_run_maker: Callable[..., Awaitable[Any]],
    session: AsyncSession,
    work_pool: Any,
):
    flow_run = await cancelling_flow_run_maker(
        state=states.Pending(), infrastructure_pid="infra-pid-1"
    )
    assert flow_run.start_time is None

    docket_name = f"test-{uuid4()}"
    with temporary_settings(
        {PREFECT_SERVER_SERVICES_CANCELLATION_CLEANUP_ENABLED: True}
    ):
        async with Docket(name=docket_name, url="memory://") as api_docket:
            async with session.begin():
                cancelling = await models.flow_runs.set_flow_run_state(
                    session=session,
                    flow_run_id=flow_run.id,
                    state=states.Cancelling(),
                    flow_policy=CoreFlowPolicy,
                )
            assert cancelling.status == responses.SetStateStatus.ACCEPT
            assert cancelling.state is not None
            await maybe_schedule_cancelling_timeout_check_for_state(
                docket=api_docket, flow_run_id=flow_run.id, state=cancelling.state
            )
            # Only what happens before the CANCELLING timeout is under test; a
            # worker would otherwise wait for the timeout check to come due.
            await api_docket.cancel(cancelling_timeout_check_key(flow_run.id))
            # Fail fast rather than have the worker below wait for a later task.
            snapshot = await api_docket.snapshot()
            assert all(task.when <= now("UTC") for task in snapshot.future)

        await _run_background_worker(docket_name)

    await session.refresh(flow_run)
    assert flow_run.state is not None
    assert flow_run.state.type == states.StateType.CANCELLED
    assert flow_run.state.message == CANCELLED_BEFORE_START_MESSAGE
    reserved = await cleanup_queue.reserve(work_pool_id=work_pool.id)
    assert reserved is not None
    assert reserved.kind == CANCELLING_TIMEOUT_TEARDOWN
    assert reserved.target == {
        "flow_run_id": str(flow_run.id),
        "infrastructure_pid": "infra-pid-1",
    }


async def test_ensure_cancelling_timeout_checks_cancels_flow_run_before_start_once(
    cleanup_queue: WorkerCleanupQueue,
    cancelling_flow_run_maker: Callable[..., Awaitable[Any]],
    session: AsyncSession,
    work_pool: Any,
):
    flow_run = await cancelling_flow_run_maker(
        state=states.Cancelling(timestamp=now("UTC") - Duration(seconds=30))
    )
    docket_name = f"test-{uuid4()}"

    for _ in range(2):
        with temporary_settings(
            {
                PREFECT_SERVER_SERVICES_CANCELLATION_CLEANUP_ENABLED: True,
                PREFECT_SERVER_SERVICES_CANCELLATION_CLEANUP_CANCELLING_TIMEOUT_SECONDS: 60,
            }
        ):
            async with Docket(name=docket_name, url="memory://") as docket:
                await ensure_cancelling_timeout_checks(
                    docket=docket, db=provide_database_interface()
                )
                # Only what happens before the CANCELLING timeout is under test; a
                # worker would otherwise wait for the timeout check to come due.
                await docket.cancel(cancelling_timeout_check_key(flow_run.id))
            await _run_background_worker(docket_name)

    await session.refresh(flow_run)
    assert flow_run.state is not None
    assert flow_run.state.message == CANCELLED_BEFORE_START_MESSAGE
    reserved = await cleanup_queue.reserve(work_pool_id=work_pool.id)
    assert reserved is not None
    assert reserved.target["infrastructure_pid"] == "infra-pid-1"
    assert await cleanup_queue.reserve(work_pool_id=work_pool.id) is None


async def test_ensure_cancelling_timeout_checks_cancels_run_whose_pid_was_recorded_late(
    cleanup_queue: WorkerCleanupQueue,
    cancelling_flow_run_maker: Callable[..., Awaitable[Any]],
    session: AsyncSession,
    work_pool: Any,
):
    flow_run = await cancelling_flow_run_maker(
        state=states.Pending(), infrastructure_pid=None
    )
    docket_name = f"test-{uuid4()}"
    with temporary_settings(
        {
            PREFECT_SERVER_SERVICES_CANCELLATION_CLEANUP_ENABLED: True,
            PREFECT_SERVER_SERVICES_CANCELLATION_CLEANUP_CANCELLING_TIMEOUT_SECONDS: 60,
        }
    ):
        async with Docket(name=docket_name, url="memory://") as api_docket:
            async with session.begin():
                cancelling = await models.flow_runs.set_flow_run_state(
                    session=session,
                    flow_run_id=flow_run.id,
                    state=states.Cancelling(),
                    flow_policy=CoreFlowPolicy,
                )
            assert cancelling.state is not None
            await maybe_schedule_cancelling_timeout_check_for_state(
                docket=api_docket, flow_run_id=flow_run.id, state=cancelling.state
            )
            # Only what happens before the CANCELLING timeout is under test; a
            # worker would otherwise wait for the timeout check to come due.
            await api_docket.cancel(cancelling_timeout_check_key(flow_run.id))
        await _run_background_worker(docket_name)
        # no infrastructure to tear down yet
        await session.refresh(flow_run)
        assert flow_run.state_type == states.StateType.CANCELLING

        # the worker records the infrastructure after the run entered CANCELLING
        await models.flow_runs.update_flow_run(
            session=session,
            flow_run_id=flow_run.id,
            flow_run=schemas.actions.FlowRunUpdate(infrastructure_pid="infra-pid-late"),
        )
        await session.commit()
        async with Docket(name=docket_name, url="memory://") as docket:
            await ensure_cancelling_timeout_checks(
                docket=docket, db=provide_database_interface()
            )
            await docket.cancel(cancelling_timeout_check_key(flow_run.id))
        await _run_background_worker(docket_name)

    await session.refresh(flow_run)
    assert flow_run.state is not None
    assert flow_run.state.message == CANCELLED_BEFORE_START_MESSAGE
    reserved = await cleanup_queue.reserve(work_pool_id=work_pool.id)
    assert reserved is not None
    assert reserved.target["infrastructure_pid"] == "infra-pid-late"


async def test_ensure_cancelling_timeout_checks_reseeds_the_same_cancellation_before_start(
    cancelling_flow_run_maker: Callable[..., Awaitable[Any]],
):
    flow_run = await cancelling_flow_run_maker(
        state=states.Cancelling(timestamp=now("UTC") - Duration(seconds=30))
    )
    cancelling_state = flow_run.state.as_state()

    scheduled: list[dict[str, tuple[datetime, dict[str, Any]]]] = []
    with temporary_settings(
        {PREFECT_SERVER_SERVICES_CANCELLATION_CLEANUP_ENABLED: True}
    ):
        async with Docket(name=f"test-{uuid4()}", url="memory://") as docket:
            await maybe_schedule_cancelling_timeout_check_for_state(
                docket=docket, flow_run_id=flow_run.id, state=cancelling_state
            )
            snapshot = await docket.snapshot()
        scheduled.append(
            {task.key: (task.when, task.kwargs) for task in snapshot.future}
        )

        async with Docket(name=f"test-{uuid4()}", url="memory://") as docket:
            await ensure_cancelling_timeout_checks(
                docket=docket, db=provide_database_interface()
            )
            snapshot = await docket.snapshot()
        scheduled.append(
            {task.key: (task.when, task.kwargs) for task in snapshot.future}
        )

    after_cancelling, after_restart = scheduled
    assert set(after_restart) == {
        cancelling_timeout_check_key(flow_run.id),
        cancel_before_start_key(flow_run.id),
    }
    cancel_key = cancel_before_start_key(flow_run.id)
    assert after_restart[cancel_key] == after_cancelling[cancel_key]


@pytest.mark.parametrize(
    "flow_run_kwargs",
    [
        {"start_time": now("UTC") - Duration(seconds=60)},
        {"infrastructure_pid": None},
    ],
    ids=["started", "no-infrastructure-pid"],
)
async def test_ensure_cancelling_timeout_checks_seeds_no_cancellation_before_start_for_runs_it_does_not_own(
    cancelling_flow_run_maker: Callable[..., Awaitable[Any]],
    flow_run_kwargs: dict[str, Any],
):
    flow_run = await cancelling_flow_run_maker(**flow_run_kwargs)

    with temporary_settings(
        {PREFECT_SERVER_SERVICES_CANCELLATION_CLEANUP_ENABLED: True}
    ):
        async with Docket(name=f"test-{uuid4()}", url="memory://") as docket:
            await ensure_cancelling_timeout_checks(
                docket=docket, db=provide_database_interface()
            )
            snapshot = await docket.snapshot()

    assert {task.key for task in snapshot.future} == {
        cancelling_timeout_check_key(flow_run.id)
    }


async def test_ensure_cancelling_timeout_checks_schedules_existing_cancelling_run(
    cancelling_flow_run_maker: Callable[..., Awaitable[Any]],
):
    state_timestamp = now("UTC") - Duration(seconds=30)
    flow_run = await cancelling_flow_run_maker(
        state=states.Cancelling(timestamp=state_timestamp)
    )

    async with Docket(name=f"test-{uuid4()}", url="memory://") as docket:
        with temporary_settings(
            {
                PREFECT_SERVER_SERVICES_CANCELLATION_CLEANUP_ENABLED: True,
                PREFECT_SERVER_SERVICES_CANCELLATION_CLEANUP_CANCELLING_TIMEOUT_SECONDS: 60,
            }
        ):
            await ensure_cancelling_timeout_checks(
                docket=docket, db=provide_database_interface()
            )
        snapshot = await docket.snapshot()

    task_keys = [task.key for task in snapshot.future]
    task_keys.extend(task.key for task in snapshot.running)
    assert cancelling_timeout_check_key(flow_run.id) in task_keys


async def test_ensure_cancelling_timeout_checks_pages_past_first_batch(
    session: AsyncSession,
    flow: Flow,
    work_pool: Any,
):
    state_timestamp = now("UTC") - Duration(seconds=30)
    flow_run_ids = []
    async with session.begin():
        for _ in range(201):
            flow_run = await models.flow_runs.create_flow_run(
                session=session,
                flow_run=schemas.core.FlowRun(
                    flow_id=flow.id,
                    state=states.Cancelling(timestamp=state_timestamp),
                    infrastructure_pid="infra-pid-1",
                    work_queue_id=work_pool.default_queue_id,
                ),
            )
            flow_run_ids.append(flow_run.id)

    async with Docket(name=f"test-{uuid4()}", url="memory://") as docket:
        with temporary_settings(
            {
                PREFECT_SERVER_SERVICES_CANCELLATION_CLEANUP_ENABLED: True,
                PREFECT_SERVER_SERVICES_CANCELLATION_CLEANUP_CANCELLING_TIMEOUT_SECONDS: 60,
            }
        ):
            await ensure_cancelling_timeout_checks(
                docket=docket, db=provide_database_interface()
            )
        snapshot = await docket.snapshot()

    task_keys = [task.key for task in snapshot.future]
    task_keys.extend(task.key for task in snapshot.running)
    assert cancelling_timeout_check_key(max(flow_run_ids)) in task_keys


async def test_monitor_cancelled_flow_runs_pages_past_first_batch(
    session: AsyncSession,
    flow: Flow,
):
    flow_run_ids = []
    async with session.begin():
        for _ in range(201):
            flow_run = await models.flow_runs.create_flow_run(
                session=session,
                flow_run=schemas.core.FlowRun(
                    flow_id=flow.id, state=states.Cancelled(), end_time=THE_PAST
                ),
            )
            flow_run_ids.append(flow_run.id)

    async with Docket(name=f"test-{uuid4()}", url="memory://") as docket:
        # the second pass checks that pending cleanups aren't enqueued again
        for _ in range(2):
            await monitor_cancelled_flow_runs(
                docket=docket, db=provide_database_interface()
            )
        snapshot = await docket.snapshot()

    scheduled_ids = [execution.args[0] for execution in snapshot.future]
    assert sorted(scheduled_ids) == sorted(flow_run_ids)


async def test_monitor_subflow_runs_pages_past_first_batch(
    session: AsyncSession,
    flow: Flow,
    cancelled_flow_run: FlowRun,
):
    subflow_run_ids = []
    async with session.begin():
        virtual_task = await models.task_runs.create_task_run(
            session=session,
            task_run=schemas.core.TaskRun(
                flow_run_id=cancelled_flow_run.id,
                task_key="a virtual task",
                dynamic_key="a virtual dynamic key",
                state=states.Running(),
            ),
        )
        for _ in range(201):
            subflow_run = await models.flow_runs.create_flow_run(
                session=session,
                flow_run=schemas.core.FlowRun(
                    flow_id=flow.id,
                    parent_task_run_id=virtual_task.id,
                    state=states.Running(),
                ),
            )
            subflow_run_ids.append(subflow_run.id)

    async with Docket(name=f"test-{uuid4()}", url="memory://") as docket:
        # the second pass checks that pending cleanups aren't enqueued again
        for _ in range(2):
            await monitor_subflow_runs(docket=docket, db=provide_database_interface())
        snapshot = await docket.snapshot()

    scheduled_ids = [execution.args[0] for execution in snapshot.future]
    assert sorted(scheduled_ids) == sorted(subflow_run_ids)


async def test_handle_cancelling_timeout_cancels_and_enqueues_cleanup(
    cleanup_queue: WorkerCleanupQueue,
    cancelling_flow_run_maker: Callable[..., Awaitable[Any]],
    session: AsyncSession,
    work_pool: Any,
):
    flow_run = await cancelling_flow_run_maker(infrastructure_pid="infra-pid-1")

    with temporary_settings(
        {PREFECT_SERVER_SERVICES_CANCELLATION_CLEANUP_CANCELLING_TIMEOUT_SECONDS: 60}
    ):
        message, timeout_state_id = await _handle_cancelling_timeout_for_run(
            flow_run, cleanup_queue
        )

    await session.refresh(flow_run)
    assert flow_run.state is not None
    assert flow_run.state.type == states.StateType.CANCELLED
    assert flow_run.state_id == timeout_state_id
    assert flow_run.state.message == CANCELLING_TIMEOUT_CANCELLED_MESSAGE
    assert flow_run.end_time is not None
    cleanup_state_id = flow_run.state_id
    assert cleanup_state_id is not None

    assert message is not None
    assert message.kind == CANCELLING_TIMEOUT_TEARDOWN
    assert message.work_pool_id == work_pool.id
    assert message.work_queue_id == work_pool.default_queue_id
    assert (
        message.idempotency_key
        == f"{CANCELLING_TIMEOUT_TEARDOWN}:{flow_run.id}:{cleanup_state_id}"
    )
    assert message.target == {
        "flow_run_id": str(flow_run.id),
        "infrastructure_pid": "infra-pid-1",
    }

    reserved = await cleanup_queue.reserve(work_pool_id=work_pool.id)
    assert reserved is not None
    assert reserved.message_id == message.message_id


async def test_handle_cancelling_timeout_skips_recent_cancelling_run(
    cleanup_queue: WorkerCleanupQueue,
    cancelling_flow_run_maker: Callable[..., Awaitable[Any]],
    session: AsyncSession,
    work_pool: Any,
):
    flow_run = await cancelling_flow_run_maker(
        state=states.Cancelling(timestamp=now("UTC") - Duration(seconds=30))
    )
    retry = RetryRecorder()

    with temporary_settings(
        {PREFECT_SERVER_SERVICES_CANCELLATION_CLEANUP_CANCELLING_TIMEOUT_SECONDS: 60}
    ):
        message, _ = await _handle_cancelling_timeout_for_run(
            flow_run, cleanup_queue, retry=retry
        )

    assert message is None
    assert retry.when is not None
    assert retry.when > now("UTC")
    await session.refresh(flow_run)
    assert flow_run.state is not None
    assert flow_run.state.type == states.StateType.CANCELLING
    assert await cleanup_queue.reserve(work_pool_id=work_pool.id) is None


@pytest.mark.parametrize(
    "state",
    [
        states.Running(timestamp=now("UTC") - Duration(seconds=120)),
        states.Cancelled(timestamp=now("UTC") - Duration(seconds=120)),
    ],
)
async def test_handle_cancelling_timeout_skips_non_cancelling_runs(
    cleanup_queue: WorkerCleanupQueue,
    cancelling_flow_run_maker: Callable[..., Awaitable[Any]],
    work_pool: Any,
    state: states.State,
):
    flow_run = await cancelling_flow_run_maker(state=state)

    with temporary_settings(
        {PREFECT_SERVER_SERVICES_CANCELLATION_CLEANUP_CANCELLING_TIMEOUT_SECONDS: 60}
    ):
        message, _ = await _handle_cancelling_timeout_for_run(flow_run, cleanup_queue)

    assert message is None
    assert await cleanup_queue.reserve(work_pool_id=work_pool.id) is None


async def test_handle_cancelling_timeout_does_not_trust_matching_user_message(
    cleanup_queue: WorkerCleanupQueue,
    cancelling_flow_run_maker: Callable[..., Awaitable[Any]],
    work_pool: Any,
):
    flow_run = await cancelling_flow_run_maker(
        state=states.Cancelled(
            timestamp=now("UTC") - Duration(seconds=120),
            message=CANCELLING_TIMEOUT_CANCELLED_MESSAGE,
        )
    )

    message, _ = await _handle_cancelling_timeout_for_run(
        flow_run,
        cleanup_queue,
        flow_run_state_id=flow_run.state_id,
        timeout_cancelled_state_id=uuid4(),
    )

    assert message is None
    assert await cleanup_queue.reserve(work_pool_id=work_pool.id) is None


async def test_handle_cancelling_timeout_skips_cleanup_when_timeout_state_is_overwritten(
    cleanup_queue: WorkerCleanupQueue,
    cancelling_flow_run_maker: Callable[..., Awaitable[Any]],
    monkeypatch: pytest.MonkeyPatch,
    session: AsyncSession,
    work_pool: Any,
):
    flow_run = await cancelling_flow_run_maker(infrastructure_pid="infra-pid-1")
    original_state_id = flow_run.state_id
    assert original_state_id is not None
    original_set_flow_run_state = models.flow_runs.set_flow_run_state

    async def set_timeout_then_stale_final_state(**kwargs: Any):
        result = await original_set_flow_run_state(**kwargs)
        await original_set_flow_run_state(
            session=kwargs["session"],
            flow_run_id=kwargs["flow_run_id"],
            state=states.Completed(),
            force=True,
        )
        return result

    monkeypatch.setattr(
        models.flow_runs,
        "set_flow_run_state",
        set_timeout_then_stale_final_state,
    )

    with temporary_settings(
        {PREFECT_SERVER_SERVICES_CANCELLATION_CLEANUP_CANCELLING_TIMEOUT_SECONDS: 60}
    ):
        message, timeout_state_id = await _handle_cancelling_timeout_for_run(
            flow_run,
            cleanup_queue,
            flow_run_state_id=original_state_id,
        )

    assert message is None
    assert await cleanup_queue.reserve(work_pool_id=work_pool.id) is None
    await session.refresh(flow_run)
    assert flow_run.state is not None
    assert flow_run.state.type == states.StateType.COMPLETED
    assert flow_run.state_id != timeout_state_id


async def test_handle_cancelling_timeout_skips_unroutable_runs(
    cleanup_queue: WorkerCleanupQueue,
    cancelling_flow_run_maker: Callable[..., Awaitable[Any]],
    caplog: pytest.LogCaptureFixture,
    session: AsyncSession,
):
    flow_run = await cancelling_flow_run_maker(use_default_work_queue=False)

    with temporary_settings(
        {PREFECT_SERVER_SERVICES_CANCELLATION_CLEANUP_CANCELLING_TIMEOUT_SECONDS: 60}
    ):
        with caplog.at_level(
            logging.WARNING, logger="prefect.server.services.cancellation_cleanup"
        ):
            message, _ = await _handle_cancelling_timeout_for_run(
                flow_run, cleanup_queue
            )

    assert message is None
    assert "unroutable flow run" in caplog.text
    assert str(flow_run.id) in caplog.text
    await session.refresh(flow_run)
    assert flow_run.state is not None
    assert flow_run.state.type == states.StateType.CANCELLED
    assert flow_run.state.message == CANCELLING_TIMEOUT_CANCELLED_MESSAGE


async def test_handle_cancelling_timeout_skips_push_pool_runs(
    cleanup_queue: WorkerCleanupQueue,
    cancelling_flow_run_maker: Callable[..., Awaitable[Any]],
    push_work_pool: Any,
    session: AsyncSession,
):
    flow_run = await cancelling_flow_run_maker(work_pool_override=push_work_pool)

    with temporary_settings(
        {PREFECT_SERVER_SERVICES_CANCELLATION_CLEANUP_CANCELLING_TIMEOUT_SECONDS: 60}
    ):
        message, _ = await _handle_cancelling_timeout_for_run(flow_run, cleanup_queue)

    assert message is None
    await session.refresh(flow_run)
    assert flow_run.state is not None
    assert flow_run.state.type == states.StateType.CANCELLED
    assert flow_run.state.message == CANCELLING_TIMEOUT_CANCELLED_MESSAGE
    assert await cleanup_queue.reserve(work_pool_id=push_work_pool.id) is None


async def test_handle_cancelling_timeout_skips_managed_pool_runs(
    cleanup_queue: WorkerCleanupQueue,
    cancelling_flow_run_maker: Callable[..., Awaitable[Any]],
    managed_work_pool: Any,
    session: AsyncSession,
):
    flow_run = await cancelling_flow_run_maker(work_pool_override=managed_work_pool)

    with temporary_settings(
        {PREFECT_SERVER_SERVICES_CANCELLATION_CLEANUP_CANCELLING_TIMEOUT_SECONDS: 60}
    ):
        message, _ = await _handle_cancelling_timeout_for_run(flow_run, cleanup_queue)

    assert message is None
    await session.refresh(flow_run)
    assert flow_run.state is not None
    assert flow_run.state.type == states.StateType.CANCELLED
    assert flow_run.state.message == CANCELLING_TIMEOUT_CANCELLED_MESSAGE
    assert await cleanup_queue.reserve(work_pool_id=managed_work_pool.id) is None


async def test_handle_cancelling_timeout_allows_missing_infrastructure_pid(
    cleanup_queue: WorkerCleanupQueue,
    cancelling_flow_run_maker: Callable[..., Awaitable[Any]],
):
    flow_run = await cancelling_flow_run_maker(infrastructure_pid=None)

    with temporary_settings(
        {PREFECT_SERVER_SERVICES_CANCELLATION_CLEANUP_CANCELLING_TIMEOUT_SECONDS: 60}
    ):
        message, _ = await _handle_cancelling_timeout_for_run(flow_run, cleanup_queue)

    assert message is not None
    assert message.target == {
        "flow_run_id": str(flow_run.id),
        "infrastructure_pid": None,
    }


async def test_handle_cancelling_timeout_is_idempotent_for_repeated_runs(
    cleanup_queue: WorkerCleanupQueue,
    cancelling_flow_run_maker: Callable[..., Awaitable[Any]],
    session: AsyncSession,
    work_pool: Any,
):
    flow_run = await cancelling_flow_run_maker()
    original_state_id = flow_run.state_id
    assert original_state_id is not None
    timeout_state_id = uuid4()

    with temporary_settings(
        {PREFECT_SERVER_SERVICES_CANCELLATION_CLEANUP_CANCELLING_TIMEOUT_SECONDS: 60}
    ):
        first, _ = await _handle_cancelling_timeout_for_run(
            flow_run,
            cleanup_queue,
            flow_run_state_id=original_state_id,
            timeout_cancelled_state_id=timeout_state_id,
        )
        second, _ = await _handle_cancelling_timeout_for_run(
            flow_run,
            cleanup_queue,
            flow_run_state_id=original_state_id,
            timeout_cancelled_state_id=timeout_state_id,
        )

    assert first is not None
    assert second is not None
    message_ids = {first.message_id, second.message_id}
    assert len(message_ids) == 1

    await session.refresh(flow_run)
    assert flow_run.state is not None
    assert flow_run.state.type == states.StateType.CANCELLED
    assert flow_run.state_id == timeout_state_id

    reserved = await cleanup_queue.reserve(work_pool_id=work_pool.id)
    assert reserved is not None
    assert reserved.message_id in message_ids
    assert await cleanup_queue.reserve(work_pool_id=work_pool.id) is None


async def test_handle_cancelling_timeout_retries_after_enqueue_failure(
    cancelling_flow_run_maker: Callable[..., Awaitable[Any]],
    monkeypatch: pytest.MonkeyPatch,
    session: AsyncSession,
    work_pool: Any,
):
    cleanup_queue = WorkerCleanupQueue()
    original_enqueue = cleanup_queue.enqueue
    failures_remaining = 1

    async def fail_once_enqueue(**kwargs: Any):
        nonlocal failures_remaining
        if failures_remaining:
            failures_remaining -= 1
            raise RuntimeError("cleanup queue unavailable")
        return await original_enqueue(**kwargs)

    monkeypatch.setattr(cleanup_queue, "enqueue", fail_once_enqueue)
    flow_run = await cancelling_flow_run_maker()
    original_state_id = flow_run.state_id
    assert original_state_id is not None
    timeout_state_id = uuid4()

    with temporary_settings(
        {PREFECT_SERVER_SERVICES_CANCELLATION_CLEANUP_CANCELLING_TIMEOUT_SECONDS: 60}
    ):
        with pytest.raises(RuntimeError, match="cleanup queue unavailable"):
            await _handle_cancelling_timeout_for_run(
                flow_run,
                cleanup_queue,
                flow_run_state_id=original_state_id,
                timeout_cancelled_state_id=timeout_state_id,
            )

        await session.refresh(flow_run)
        assert flow_run.state is not None
        assert flow_run.state.type == states.StateType.CANCELLING
        assert flow_run.state_id == original_state_id
        assert await cleanup_queue.reserve(work_pool_id=work_pool.id) is None

        message, _ = await _handle_cancelling_timeout_for_run(
            flow_run,
            cleanup_queue,
            flow_run_state_id=original_state_id,
            timeout_cancelled_state_id=timeout_state_id,
        )

    await session.refresh(flow_run)
    assert flow_run.state is not None
    assert flow_run.state.type == states.StateType.CANCELLED
    assert flow_run.state.message == CANCELLING_TIMEOUT_CANCELLED_MESSAGE
    cleanup_state_id = flow_run.state_id
    assert cleanup_state_id == timeout_state_id

    assert message is not None
    assert message.idempotency_key == (
        f"{CANCELLING_TIMEOUT_TEARDOWN}:{flow_run.id}:{cleanup_state_id}"
    )
    reserved = await cleanup_queue.reserve(work_pool_id=work_pool.id)
    assert reserved is not None
    assert reserved.message_id == message.message_id


async def test_handle_cancelling_timeout_uses_state_id_for_retry_attempts(
    cleanup_queue: WorkerCleanupQueue,
    cancelling_flow_run_maker: Callable[..., Awaitable[Any]],
    work_pool: Any,
):
    flow_run = await cancelling_flow_run_maker(infrastructure_pid="infra-pid-1")
    db = provide_database_interface()
    first_cancelling_state_id = flow_run.state_id
    assert first_cancelling_state_id is not None

    with temporary_settings(
        {PREFECT_SERVER_SERVICES_CANCELLATION_CLEANUP_CANCELLING_TIMEOUT_SECONDS: 60}
    ):
        first_message, _ = await _handle_cancelling_timeout_for_run(
            flow_run,
            cleanup_queue,
            flow_run_state_id=first_cancelling_state_id,
        )

    assert first_message is not None
    async with db.session_context() as first_state_session:
        first_retry_flow_run = await models.flow_runs.read_flow_run(
            session=first_state_session, flow_run_id=flow_run.id
        )
    assert first_retry_flow_run is not None
    first_cleanup_state_id = first_retry_flow_run.state_id
    assert first_cleanup_state_id is not None
    assert (
        first_message.idempotency_key == f"{CANCELLING_TIMEOUT_TEARDOWN}:"
        f"{flow_run.id}:{first_cleanup_state_id}"
    )

    first_reservation = await cleanup_queue.reserve(work_pool_id=work_pool.id)
    assert first_reservation is not None
    await cleanup_queue.ack(
        work_pool_id=work_pool.id,
        message_id=first_reservation.message_id,
        reservation_token=first_reservation.reservation_token,
    )

    async with db.session_context(begin_transaction=True) as retry_session:
        await models.flow_runs.update_flow_run(
            session=retry_session,
            flow_run_id=flow_run.id,
            flow_run=schemas.actions.FlowRunUpdate(infrastructure_pid="infra-pid-2"),
        )
        retry_state_result = await models.flow_runs.set_flow_run_state(
            session=retry_session,
            flow_run_id=flow_run.id,
            state=states.Cancelling(timestamp=now("UTC") - Duration(seconds=120)),
            force=True,
        )

    assert retry_state_result.state is not None
    assert retry_state_result.state.id is not None

    with temporary_settings(
        {PREFECT_SERVER_SERVICES_CANCELLATION_CLEANUP_CANCELLING_TIMEOUT_SECONDS: 60}
    ):
        second_message, _ = await _handle_cancelling_timeout_for_run(
            flow_run,
            cleanup_queue,
            flow_run_state_id=retry_state_result.state.id,
        )

    assert second_message is not None
    async with db.session_context() as second_state_session:
        second_retry_flow_run = await models.flow_runs.read_flow_run(
            session=second_state_session, flow_run_id=flow_run.id
        )
    assert second_retry_flow_run is not None
    second_cleanup_state_id = second_retry_flow_run.state_id
    assert second_cleanup_state_id is not None
    assert second_message.message_id != first_message.message_id
    assert (
        second_message.idempotency_key == f"{CANCELLING_TIMEOUT_TEARDOWN}:"
        f"{flow_run.id}:{second_cleanup_state_id}"
    )
    assert second_message.target == {
        "flow_run_id": str(flow_run.id),
        "infrastructure_pid": "infra-pid-2",
    }

    second_reservation = await cleanup_queue.reserve(work_pool_id=work_pool.id)
    assert second_reservation is not None
    assert second_reservation.message_id == second_message.message_id


async def test_cancel_flow_run_before_start_cancels_run_and_enqueues_teardown(
    cleanup_queue: WorkerCleanupQueue,
    cancelling_flow_run_maker: Callable[..., Awaitable[Any]],
    session: AsyncSession,
    work_pool: Any,
):
    flow_run = await cancelling_flow_run_maker(
        state=states.Cancelling(timestamp=now("UTC") - Duration(seconds=30)),
        infrastructure_pid="infra-pid-1",
    )
    cancelling_state_id = flow_run.state_id
    assert flow_run.start_time is None

    message = await _cancel_flow_run_before_start(flow_run, cleanup_queue)

    await session.refresh(flow_run)
    assert flow_run.state is not None
    assert flow_run.state.type == states.StateType.CANCELLED
    assert flow_run.state.message == CANCELLED_BEFORE_START_MESSAGE
    assert flow_run.end_time is not None

    assert message is not None
    assert message.kind == CANCELLING_TIMEOUT_TEARDOWN
    assert message.work_pool_id == work_pool.id
    assert message.work_queue_id == work_pool.default_queue_id
    assert message.idempotency_key == (
        f"{CANCELLING_TIMEOUT_TEARDOWN}:before-start:{flow_run.id}:{cancelling_state_id}"
    )
    assert message.message_id == uuid5(
        NAMESPACE_URL, f"prefect:{message.idempotency_key}"
    )
    assert message.target == {
        "flow_run_id": str(flow_run.id),
        "infrastructure_pid": "infra-pid-1",
    }
    reserved = await cleanup_queue.reserve(work_pool_id=work_pool.id)
    assert reserved is not None
    assert reserved.message_id == message.message_id


@pytest.mark.parametrize(
    "flow_run_kwargs",
    [
        {"start_time": now("UTC") - Duration(seconds=60)},
        {"infrastructure_pid": None},
        {"state": states.Pending(timestamp=now("UTC") - Duration(seconds=30))},
    ],
    ids=["started", "no-infrastructure-pid", "not-cancelling"],
)
async def test_cancel_flow_run_before_start_skips_runs_it_does_not_own(
    cleanup_queue: WorkerCleanupQueue,
    cancelling_flow_run_maker: Callable[..., Awaitable[Any]],
    session: AsyncSession,
    work_pool: Any,
    flow_run_kwargs: dict[str, Any],
):
    flow_run = await cancelling_flow_run_maker(
        **{
            "state": states.Cancelling(timestamp=now("UTC") - Duration(seconds=30)),
            **flow_run_kwargs,
        }
    )
    state_id = flow_run.state_id

    message = await _cancel_flow_run_before_start(flow_run, cleanup_queue)

    assert message is None
    assert await cleanup_queue.reserve(work_pool_id=work_pool.id) is None
    await session.refresh(flow_run)
    assert flow_run.state_id == state_id


async def test_cancel_flow_run_before_start_skips_run_in_a_new_state(
    cleanup_queue: WorkerCleanupQueue,
    cancelling_flow_run_maker: Callable[..., Awaitable[Any]],
    work_pool: Any,
):
    flow_run = await cancelling_flow_run_maker(
        state=states.Cancelling(timestamp=now("UTC") - Duration(seconds=30))
    )

    message = await _cancel_flow_run_before_start(
        flow_run, cleanup_queue, flow_run_state_id=uuid4()
    )

    assert message is None
    assert await cleanup_queue.reserve(work_pool_id=work_pool.id) is None


async def test_cancel_flow_run_before_start_skips_deleted_run(
    cleanup_queue: WorkerCleanupQueue,
    cancelling_flow_run_maker: Callable[..., Awaitable[Any]],
    session: AsyncSession,
    work_pool: Any,
):
    flow_run = await cancelling_flow_run_maker(
        state=states.Cancelling(timestamp=now("UTC") - Duration(seconds=30))
    )
    async with session.begin():
        assert await models.flow_runs.delete_flow_run(
            session=session, flow_run_id=flow_run.id
        )

    message = await _cancel_flow_run_before_start(flow_run, cleanup_queue)

    assert message is None
    assert await cleanup_queue.reserve(work_pool_id=work_pool.id) is None


@pytest.mark.parametrize("routing", ["unroutable", "push", "managed"])
async def test_cancel_flow_run_before_start_leaves_runs_without_workers_cancelling(
    cleanup_queue: WorkerCleanupQueue,
    cancelling_flow_run_maker: Callable[..., Awaitable[Any]],
    push_work_pool: Any,
    managed_work_pool: Any,
    caplog: pytest.LogCaptureFixture,
    session: AsyncSession,
    routing: str,
):
    recent_state = states.Cancelling(timestamp=now("UTC") - Duration(seconds=30))
    if routing == "unroutable":
        flow_run = await cancelling_flow_run_maker(
            state=recent_state, use_default_work_queue=False
        )
    else:
        flow_run = await cancelling_flow_run_maker(
            state=recent_state,
            work_pool_override=(
                push_work_pool if routing == "push" else managed_work_pool
            ),
        )

    with caplog.at_level(
        logging.INFO, logger="prefect.server.services.cancellation_cleanup"
    ):
        message = await _cancel_flow_run_before_start(flow_run, cleanup_queue)

    assert message is None
    await session.refresh(flow_run)
    assert flow_run.state_type == states.StateType.CANCELLING
    # the CANCELLING timeout check reports why no worker cleanup was enqueued
    assert [
        record
        for record in caplog.records
        if record.name == "prefect.server.services.cancellation_cleanup"
    ] == []


async def test_handle_cancelling_timeout_skips_run_cancelled_before_it_started(
    cleanup_queue: WorkerCleanupQueue,
    cancelling_flow_run_maker: Callable[..., Awaitable[Any]],
    session: AsyncSession,
    work_pool: Any,
):
    flow_run = await cancelling_flow_run_maker()
    cancelling_state_id = flow_run.state_id
    early_message = await _cancel_flow_run_before_start(flow_run, cleanup_queue)
    assert early_message is not None

    with temporary_settings(
        {PREFECT_SERVER_SERVICES_CANCELLATION_CLEANUP_CANCELLING_TIMEOUT_SECONDS: 60}
    ):
        timeout_message, _ = await _handle_cancelling_timeout_for_run(
            flow_run, cleanup_queue, flow_run_state_id=cancelling_state_id
        )

    assert timeout_message is None
    await session.refresh(flow_run)
    assert flow_run.state is not None
    assert flow_run.state.message == CANCELLED_BEFORE_START_MESSAGE
    reserved = await cleanup_queue.reserve(work_pool_id=work_pool.id)
    assert reserved is not None
    assert reserved.message_id == early_message.message_id
    assert await cleanup_queue.reserve(work_pool_id=work_pool.id) is None


async def test_cancelling_state_replaces_the_cancellation_before_start_of_an_earlier_one():
    flow_run_id = uuid4()
    earlier = states.Cancelling(id=uuid4(), timestamp=now("UTC") - Duration(seconds=10))
    later = states.Cancelling(id=uuid4(), timestamp=now("UTC") - Duration(seconds=5))
    async with Docket(name=f"test-{uuid4()}", url="memory://") as docket:
        with temporary_settings(
            {PREFECT_SERVER_SERVICES_CANCELLATION_CLEANUP_ENABLED: True}
        ):
            for state in (earlier, later):
                await maybe_schedule_cancelling_timeout_check_for_state(
                    docket=docket, flow_run_id=flow_run_id, state=state
                )
        snapshot = await docket.snapshot()

    cancellations = [
        task
        for task in snapshot.future
        if task.key == cancel_before_start_key(flow_run_id)
    ]
    assert [task.kwargs["flow_run_state_id"] for task in cancellations] == [later.id]


async def test_cancel_flow_run_before_start_is_retried_when_enqueue_fails(
    cleanup_queue: WorkerCleanupQueue,
    cancelling_flow_run_maker: Callable[..., Awaitable[Any]],
    monkeypatch: pytest.MonkeyPatch,
    session: AsyncSession,
    work_pool: Any,
):
    flow_run = await cancelling_flow_run_maker()
    enqueue = cleanup_queue.enqueue
    attempts = 0

    async def enqueue_failing_once(**kwargs: Any) -> Any:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("cleanup queue unavailable")
        return await enqueue(**kwargs)

    monkeypatch.setattr(cleanup_queue, "enqueue", enqueue_failing_once)
    docket_name = f"test-{uuid4()}"
    async with Docket(name=docket_name, url="memory://") as docket:
        await docket.add(
            cancel_flow_run_before_start, key=cancel_before_start_key(flow_run.id)
        )(flow_run_id=flow_run.id, flow_run_state_id=flow_run.state_id)

    await _run_background_worker(docket_name)

    assert attempts == 2
    await session.refresh(flow_run)
    assert flow_run.state is not None
    assert flow_run.state.message == CANCELLED_BEFORE_START_MESSAGE
    reserved = await cleanup_queue.reserve(work_pool_id=work_pool.id)
    assert reserved is not None
    assert reserved.target["infrastructure_pid"] == "infra-pid-1"


async def test_cancel_flow_run_before_start_enqueues_nothing_when_cancelled_is_not_accepted(
    cleanup_queue: WorkerCleanupQueue,
    cancelling_flow_run_maker: Callable[..., Awaitable[Any]],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    session: AsyncSession,
    work_pool: Any,
):
    # No orchestration rule rejects CANCELLING -> CANCELLED today, so a stub
    # stands in for one that would.
    async def reject(**_: Any) -> responses.OrchestrationResult:
        return responses.OrchestrationResult(
            state=None,
            status=responses.SetStateStatus.REJECT,
            details=responses.StateRejectDetails(reason="rejected for the test"),
        )

    flow_run = await cancelling_flow_run_maker()
    cancelling_state_id = flow_run.state_id
    monkeypatch.setattr(models.flow_runs, "set_flow_run_state", reject)

    with caplog.at_level(
        logging.INFO, logger="prefect.server.services.cancellation_cleanup"
    ):
        message = await _cancel_flow_run_before_start(flow_run, cleanup_queue)

    assert message is None
    assert await cleanup_queue.reserve(work_pool_id=work_pool.id) is None
    await session.refresh(flow_run)
    assert flow_run.state_id == cancelling_state_id
    assert [
        record.levelno
        for record in caplog.records
        if record.name == "prefect.server.services.cancellation_cleanup"
    ] == [logging.INFO]


async def test_cancel_flow_run_before_start_logs_the_enqueued_teardown(
    cleanup_queue: WorkerCleanupQueue,
    cancelling_flow_run_maker: Callable[..., Awaitable[Any]],
    caplog: pytest.LogCaptureFixture,
):
    flow_run = await cancelling_flow_run_maker()

    with caplog.at_level(
        logging.INFO, logger="prefect.server.services.cancellation_cleanup"
    ):
        message = await _cancel_flow_run_before_start(flow_run, cleanup_queue)

    assert message is not None
    records = [
        record
        for record in caplog.records
        if record.name == "prefect.server.services.cancellation_cleanup"
    ]
    assert [record.levelno for record in records] == [logging.INFO]
    assert str(message.message_id) in records[0].getMessage()


@pytest.mark.parametrize("state_constructor", NON_TERMINAL_STATE_CONSTRUCTORS.items())
async def test_cancel_child_task_runs_cancels_nonterminal_tasks(
    session: AsyncSession,
    cancelled_flow_run: FlowRun,
    orphaned_task_run_maker: Callable[..., Any],
    state_constructor: tuple[states.StateType, Any],
):
    """Test that cancel_child_task_runs cancels non-terminal task runs."""
    orphaned_task_run = await orphaned_task_run_maker(
        cancelled_flow_run, state_constructor[1]
    )
    assert cancelled_flow_run.state.type == "CANCELLED"
    assert orphaned_task_run.state.type == state_constructor[0]

    # Call the docket task function directly
    db = provide_database_interface()
    await cancel_child_task_runs(cancelled_flow_run.id, db=db)

    await session.refresh(orphaned_task_run)
    assert orphaned_task_run.state.type == "CANCELLED"


@pytest.mark.parametrize("state_constructor", NON_TERMINAL_STATE_CONSTRUCTORS.items())
async def test_cancel_subflow_run_cancels_nonterminal_subflows(
    session: AsyncSession,
    cancelled_flow_run: FlowRun,
    orphaned_subflow_run_maker: Callable[..., Any],
    state_constructor: tuple[states.StateType, Any],
):
    """Test that cancel_subflow_run cancels non-terminal subflow runs."""
    orphaned_subflow_run = await orphaned_subflow_run_maker(
        cancelled_flow_run, state_constructor[1]
    )
    assert cancelled_flow_run.state.type == "CANCELLED"
    assert orphaned_subflow_run.state.type == state_constructor[0]

    # Call the docket task function directly
    db = provide_database_interface()
    await cancel_subflow_run(orphaned_subflow_run.id, db=db)

    await session.refresh(orphaned_subflow_run)
    assert orphaned_subflow_run.state.type == "CANCELLED"


@pytest.mark.parametrize(
    "state_constructor",
    [
        (state_type, state_constructor)
        for state_type, state_constructor in NON_TERMINAL_STATE_CONSTRUCTORS.items()
        if state_type != states.StateType.CANCELLING
    ],
)
async def test_cancel_subflow_run_sets_cancelling_for_deployment_subflows(
    session: AsyncSession,
    cancelled_flow_run: FlowRun,
    orphaned_subflow_run_from_deployment_maker: Callable[..., Any],
    state_constructor: tuple[states.StateType, Any],
):
    """Test that subflows from deployments get CANCELLING state (not CANCELLED)."""
    orphaned_subflow_run = await orphaned_subflow_run_from_deployment_maker(
        cancelled_flow_run, state_constructor[1]
    )
    assert cancelled_flow_run.state.type == "CANCELLED"
    assert orphaned_subflow_run.state.type == state_constructor[0]

    db = provide_database_interface()
    async with Docket(name=f"test-{uuid4()}", url="memory://") as docket:
        with temporary_settings(
            {PREFECT_SERVER_SERVICES_CANCELLATION_CLEANUP_ENABLED: True}
        ):
            await cancel_subflow_run(orphaned_subflow_run.id, docket=docket, db=db)

        snapshot = await docket.snapshot()

    await session.refresh(orphaned_subflow_run)
    # Subflows from deployments should be set to CANCELLING (not CANCELLED)
    # so the infrastructure can be properly cleaned up
    assert orphaned_subflow_run.state.type == "CANCELLING"
    assert orphaned_subflow_run.state_id is not None

    task_keys = {task.key for task in snapshot.future}
    task_keys.update(task.key for task in snapshot.running)
    assert cancelling_timeout_check_key(orphaned_subflow_run.id) in task_keys


async def test_cancel_subflow_run_leaves_cancelling_deployment_subflows_alone(
    session: AsyncSession,
    cancelled_flow_run: FlowRun,
    orphaned_subflow_run_from_deployment_maker: Callable[..., Any],
):
    """Already-cancelling deployment subflows should keep their timeout deadline."""
    orphaned_subflow_run = await orphaned_subflow_run_from_deployment_maker(
        cancelled_flow_run, states.Cancelling
    )
    initial_state_id = orphaned_subflow_run.state_id
    assert initial_state_id is not None

    db = provide_database_interface()
    async with Docket(name=f"test-{uuid4()}", url="memory://") as docket:
        with temporary_settings(
            {PREFECT_SERVER_SERVICES_CANCELLATION_CLEANUP_ENABLED: True}
        ):
            await cancel_subflow_run(orphaned_subflow_run.id, docket=docket, db=db)

        snapshot = await docket.snapshot()

    await session.refresh(orphaned_subflow_run)
    assert orphaned_subflow_run.state.type == "CANCELLING"
    assert orphaned_subflow_run.state_id == initial_state_id

    task_keys = {task.key for task in snapshot.future}
    task_keys.update(task.key for task in snapshot.running)
    assert cancelling_timeout_check_key(orphaned_subflow_run.id) not in task_keys


@pytest.mark.parametrize("state_constructor", TERMINAL_STATE_CONSTRUCTORS.items())
async def test_cancel_child_task_runs_leaves_terminal_tasks_alone(
    session: AsyncSession,
    cancelled_flow_run: FlowRun,
    orphaned_task_run_maker: Callable[..., Any],
    state_constructor: tuple[states.StateType, Any],
):
    """Test that cancel_child_task_runs doesn't modify terminal task runs."""
    orphaned_task_run = await orphaned_task_run_maker(
        cancelled_flow_run, state_constructor[1]
    )
    assert cancelled_flow_run.state.type == "CANCELLED"
    assert orphaned_task_run.state.type == state_constructor[0]

    # Call the docket task function directly
    db = provide_database_interface()
    await cancel_child_task_runs(cancelled_flow_run.id, db=db)

    await session.refresh(orphaned_task_run)
    # Terminal states should not be changed
    assert orphaned_task_run.state.type == state_constructor[0]


@pytest.mark.parametrize("state_constructor", TERMINAL_STATE_CONSTRUCTORS.items())
async def test_cancel_subflow_run_leaves_terminal_subflows_alone(
    session: AsyncSession,
    cancelled_flow_run: FlowRun,
    orphaned_subflow_run_maker: Callable[..., Any],
    state_constructor: tuple[states.StateType, Any],
):
    """Test that cancel_subflow_run doesn't modify terminal subflow runs."""
    orphaned_subflow_run = await orphaned_subflow_run_maker(
        cancelled_flow_run, state_constructor[1]
    )
    assert cancelled_flow_run.state.type == "CANCELLED"
    assert orphaned_subflow_run.state.type == state_constructor[0]

    # Call the docket task function directly
    db = provide_database_interface()
    await cancel_subflow_run(orphaned_subflow_run.id, db=db)

    await session.refresh(orphaned_subflow_run)
    # Terminal states should not be changed
    assert orphaned_subflow_run.state.type == state_constructor[0]


async def test_cancel_subflow_run_ignores_non_cancelled_parents(
    session: AsyncSession,
    flow: Flow,
):
    """Test that cancel_subflow_run doesn't cancel subflows whose parent isn't cancelled."""
    # Create a running parent flow run (not cancelled)
    async with session.begin():
        parent_flow = await models.flow_runs.create_flow_run(
            session=session,
            flow_run=schemas.core.FlowRun(
                flow_id=flow.id,
                state=states.Running(),
            ),
        )
        virtual_task = await models.task_runs.create_task_run(
            session=session,
            task_run=schemas.core.TaskRun(
                flow_run_id=parent_flow.id,
                task_key="virtual task",
                dynamic_key="dynamic key",
                state=states.Running(),
            ),
        )
        subflow = await models.flow_runs.create_flow_run(
            session=session,
            flow_run=schemas.core.FlowRun(
                flow_id=flow.id,
                parent_task_run_id=virtual_task.id,
                state=states.Running(),
            ),
        )

    assert parent_flow.state.type == "RUNNING"
    assert subflow.state.type == "RUNNING"

    # Call the docket task function directly
    db = provide_database_interface()
    await cancel_subflow_run(subflow.id, db=db)

    await session.refresh(subflow)
    # Subflow should still be running since parent isn't cancelled
    assert subflow.state.type == "RUNNING"


async def test_cancel_child_task_runs_handles_deleted_flow_run(
    session: AsyncSession,
    flow: Flow,
):
    """Test that cancel_child_task_runs handles the case where flow run was deleted."""
    from uuid import uuid4

    # Call with a non-existent flow run ID - should not raise
    db = provide_database_interface()
    await cancel_child_task_runs(uuid4(), db=db)


async def test_cancel_subflow_run_handles_deleted_flow_run(
    session: AsyncSession,
    flow: Flow,
):
    """Test that cancel_subflow_run handles the case where flow run was deleted."""
    from uuid import uuid4

    # Call with a non-existent flow run ID - should not raise
    db = provide_database_interface()
    await cancel_subflow_run(uuid4(), db=db)
