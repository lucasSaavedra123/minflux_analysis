import numpy as np



DELTA_T = 0.0003
DIMENSION = 2
R = 1/6
SEGMENT_LENGTH = 500

def equation_free(x, D, LOCALIZATION_PRECISION):
    TERM_1 = 2*DIMENSION*D*DELTA_T*(x-(2*R))
    TERM_2 = 2*DIMENSION*(LOCALIZATION_PRECISION**2)
    return TERM_1 + TERM_2

def equation_hop(x, DM, DU, L_HOP, LOCALIZATION_PRECISION):
    TERM_1_1_1 = (DU-DM)/DU
    TERM_1_1_2 = (L_HOP**2)/(6*DIMENSION*x*DELTA_T)
    TERM_1_1_3 = 1 - (np.exp(-((12*DU*x*DELTA_T)/(L_HOP**2))))

    TERM_1_1 = 2*DIMENSION*DELTA_T
    TERM_1_2 = DM + (TERM_1_1_1*TERM_1_1_2*TERM_1_1_3)
    TERM_1_3 = (x-(2*R))
    TERM_1 = TERM_1_1 * TERM_1_2 * TERM_1_3

    TERM_2 = 2*DIMENSION*(LOCALIZATION_PRECISION**2)
    return TERM_1 + TERM_2

def equation_confined(x, DU, L_HOP, LOCALIZATION_PRECISION):
    return equation_hop(x, 0, DU, L_HOP, LOCALIZATION_PRECISION)

X = np.linspace(1, int(0.005/DELTA_T), 100)
PARAMETERS = [1.32681384e+04, 2.94661700e+04, 3.25144280e+01, 7.99613462e+00]
"""
Y_CONFINED = equation_confined(X,PARAMETERS[0], PARAMETERS[1], PARAMETERS[2])

np.savetxt('X', X*DELTA_T)
np.savetxt('Y_CONFINED', Y_CONFINED)
"""
"""
Y_FREE = equation_free(X,PARAMETERS[0], PARAMETERS[1])

np.savetxt('X', X*DELTA_T)
np.savetxt('Y_FREE', Y_FREE)

"""
Y_HOP = equation_hop(X,PARAMETERS[0], PARAMETERS[1], PARAMETERS[2], PARAMETERS[3])
Y_DU = X * ((Y_HOP[1] - Y_HOP[0])/(X[1] - X[0])) #equation_hop(X, PARAMETERS[1], PARAMETERS[1], PARAMETERS[2], PARAMETERS[3])
Y_DM = equation_hop(X, PARAMETERS[0], PARAMETERS[0], PARAMETERS[2], PARAMETERS[3])

OFFSET = Y_DU[0] - Y_HOP[0]
Y_DU -= OFFSET

OFFSET = Y_DM[-1] - Y_HOP[-1]
Y_DM -= OFFSET

# import matplotlib.pyplot as plt
# plt.plot(X[:25], Y_HOP[:25])
# plt.plot(X[:25], Y_DU[:25])
# plt.show()

np.savetxt('X', X*DELTA_T)
np.savetxt('Y_HOP', Y_HOP)
np.savetxt('Y_DU', Y_DU)
np.savetxt('Y_DM', Y_DM)
