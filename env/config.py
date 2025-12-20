# env/config.py
ENV_KWARGS = dict(
    start_balance=10_000.0,
    fee_per_contract=0.001,
    spread=0.02,
    strike_step=100.0,
    decision_offset_minutes=60,
    trade_penalty_coef=0.001,
    pricing_k=5.0,
    pricing_bias=0.0,
    pricing_noise_std=0.06,
)
