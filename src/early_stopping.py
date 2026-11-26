"""验证准确率早停规则，供训练和历史回放共用。"""

from decimal import Decimal


def validate_early_stopping(settings):
    if settings is None:
        return None
    if not isinstance(settings, dict) or settings.keys() != {"min_epochs", "patience", "min_delta"}:
        raise ValueError("early_stopping 必须包含 min_epochs、patience 和 min_delta")
    for name in ("min_epochs", "patience"):
        if type(settings[name]) is not int or settings[name] < 1:
            raise ValueError(f"early_stopping.{name} 必须是正整数")
    delta = settings["min_delta"]
    if type(delta) not in (int, float) or not 0 <= delta < 1:
        raise ValueError("early_stopping.min_delta 必须位于 [0, 1)")
    return dict(settings, min_delta=float(delta))


def new_early_stopping_state():
    return {"reference_accuracy": None, "wait": 0, "stopped_epoch": None}


def update_early_stopping(state, settings, epoch, validation_accuracy):
    if state["stopped_epoch"] is not None:
        raise ValueError("早停触发后不能再记录新轮次")
    if type(validation_accuracy) not in (int, float) or not 0 <= validation_accuracy <= 1:
        raise ValueError("早停监控的验证准确率必须位于 [0, 1]")
    reference = state["reference_accuracy"]
    if epoch <= settings["min_epochs"]:
        state["reference_accuracy"] = validation_accuracy if reference is None else max(reference, validation_accuracy)
        return False
    # 用记录的十进制数比较，恰好等于阈值时不计为超过。
    improvement = Decimal(str(validation_accuracy)) - Decimal(str(reference))
    if improvement > Decimal(str(settings["min_delta"])):
        state.update(reference_accuracy=validation_accuracy, wait=0)
        return False
    state["wait"] += 1
    if state["wait"] == settings["patience"]:
        state["stopped_epoch"] = epoch
        return True
    return False


def replay_early_stopping(settings, history):
    state = new_early_stopping_state()
    for epoch, row in enumerate(history, start=1):
        if row["epoch"] != epoch:
            raise ValueError("早停记录的轮次必须从 1 开始连续排列")
        update_early_stopping(state, settings, epoch, row["val_accuracy"])
    return state
