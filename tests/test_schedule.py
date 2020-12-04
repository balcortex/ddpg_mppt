from src.schedule import ConstantSchedule, LinearSchedule


def test_constant_schedule():
    schedule = ConstantSchedule(epsilon=1.0)
    assert schedule() == 1.0
    assert schedule.set_current_step(10) == 1.0
    assert schedule.step() == 1.0
    assert schedule.step(steps=10) == 1.0
    assert schedule.reset() == 1.0


def test_linear_schedule():
    schedule = LinearSchedule(
        eps_start=1.0,
        eps_final=0.1,
        max_steps=100,
    )
    assert schedule() == 1.0
    assert schedule.step() == 1.0 - 1 / 100
    assert schedule.step(steps=4) == 1.0 - 5 / 100
    assert schedule.reset() == 1.0
    assert schedule() == 1.0
    assert schedule.set_current_step(step=10) == 1.0 - 0.1
    assert schedule() == 1.0 - 0.1
    assert schedule.set_current_step(step=101) == 0.1
    assert schedule() == 0.1
    assert schedule.step() == 0.1
    assert schedule.current_step == 102
