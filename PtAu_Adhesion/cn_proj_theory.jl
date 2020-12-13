using Formatting: printfmtln
Eau = 78.9 * 10^9 # elastic modulus of gold: N/m^2
νau = 0.415 # poisson's ratio of gold
Est = Eau / (1 - νau^2) # reduced elastic modulus of the gold substrate: N/m^2
K = 4/3 * Est # effective spring constant of the system
γpt = 1.6 # surface energy of platinum: J/m^2
γau = 0.75 # surface energy of gold: J/m^2
w = γpt + γau # combined surface energy of system
R = 1 * 10^-9 # radius of tip: m
a = 3.81 * 10^-10 # radius of the 'cylindrical-punch' indenter approximation of the spherical face
printfmtln("Pull-Off Force JKR: {:.1f} nN", ((-3/2 * w * pi * R)*10^9))
printfmtln("Pull-Off Force K: {:.1f} nN", (-sqrt(8 * pi * Est * w * a^3)*10^9))
printfmtln("Contact Radius: {:.2f} Å", (((R/K * 6 * w * pi * R)^(1/3))*10^10))
printfmtln("Pull-Off Contact Radius: {:.2f} Å", ((R/K * (-3/2 * w * pi * R + 3 * w * pi * R))^(1/3)*10^10))