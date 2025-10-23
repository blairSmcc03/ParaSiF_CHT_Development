import math

def sigma(kappa1, alpha1, kappa2, alpha2):
    return (kappa1/math.sqrt(alpha1)) / (kappa2/math.sqrt(alpha2))

def raleighNum(rho, g, mu, alpha):
    numerator = g*rho

    denominator = mu*alpha*1.0*1.2

    return numerator/denominator

def PrandtlNum():
    return 0.7

def grashofNum(rho, g, mu, alpha):
    return PrandtlNum()*raleighNum(rho, g, mu, alpha)


## Solid
kappaS = 1600
cS = 500
rhoS = 8000
alphaS = kappaS/(cS*rhoS)

# Fluid
cF = 1000
muF = 7e-4
kappaF = cF*muF/PrandtlNum()
M = 28.97e-3
p = 1e5

rhoF = p*M/(8.314*273.15)

print("rho fluid: {:f}".format(rhoF))

alphaF = kappaF/(cF*rhoF)
print("kappa fluid: {:f}".format(kappaF))
print("kS/kF: {:f}".format(kappaS/kappaF))
print("alphaS/alphaF: {:f}".format(alphaS/alphaF))


g = 9.8

print("Grashof number: {:f}".format(grashofNum(rhoF, g, muF, alphaF)))

print("Sigma: {:f}".format(sigma(kappaF, alphaF, kappaS, alphaS)))


print((83.14*10**2)/8.314)