PUMF_TO_NAICS18 = {
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 7,
    7: 8,
    8: 9,
    9: 15,
    10: 10,
    11: 10,
    12: 11,
    13: 12,
    14: 12,
    15: 13,
    16: 14,
    17: 15,
    18: 16,
    19: 17,
    20: 18,
}
NAICS18_TO_PUMF = {}
for pumf, naics18 in PUMF_TO_NAICS18.items():
    if naics18 not in NAICS18_TO_PUMF:
        NAICS18_TO_PUMF[naics18] = []
    NAICS18_TO_PUMF[naics18].append(pumf)
VALID_NAICS_18_CODES = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
