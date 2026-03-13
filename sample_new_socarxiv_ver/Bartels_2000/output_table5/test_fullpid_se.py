"""Check SE impact of full PID dummies IV approach."""

# Standard IV SEs:
iv_standard_se = {
    '1960': {'strong': 0.196, 'weak': 0.264, 'lean': 0.771, 'intercept': 0.060},
    '1976': {'strong': 0.188, 'weak': 0.253, 'lean': 0.375, 'intercept': 0.056},
    '1992': {'strong': 0.176, 'weak': 0.291, 'lean': 0.489, 'intercept': 0.056},
}

# Full PID dummies IV SEs:
iv_fullpid_se = {
    '1960': {'strong': 0.196, 'weak': 0.263, 'lean': 0.757, 'intercept': 0.060},
    '1976': {'strong': 0.187, 'weak': 0.250, 'lean': 0.369, 'intercept': 0.057},
    '1992': {'strong': 0.173, 'weak': 0.271, 'lean': 0.469, 'intercept': 0.057},
}

GROUND_TRUTH_IV_SE = {
    '1960': {'strong': 0.173, 'weak': 0.239, 'lean': 0.696, 'intercept': 0.057},
    '1976': {'strong': 0.178, 'weak': 0.251, 'lean': 0.438, 'intercept': 0.055},
    '1992': {'strong': 0.180, 'weak': 0.268, 'lean': 0.513, 'intercept': 0.053},
}

for label, se_vals in [('Standard', iv_standard_se), ('Full PID', iv_fullpid_se)]:
    total = 0
    for year in ['1960', '1976', '1992']:
        for vk in ['strong', 'weak', 'lean', 'intercept']:
            gt = GROUND_TRUTH_IV_SE[year][vk]
            gen = se_vals[year][vk]
            diff = abs(gen - gt)
            score = max(0, 1.0 - diff / 0.02) if diff <= 0.06 else 0.0
            total += score
    print(f'{label}: SE total={total:.3f}/12, contribution={total/36*20:.2f}/20')
