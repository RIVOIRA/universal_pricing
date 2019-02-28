from LocalVolatilityGenerator import UnitLocalVolatilityGenerator
import numpy as np

x = np.linspace(-1, 1, 5)
y = np.linspace(0., 1, 4)
alpha = 0.25
gamma = 0.5
my_loc_surf_gen = UnitLocalVolatilityGenerator(alpha, gamma, 123456)

max_degree = 3
random_loc_surf1 = my_loc_surf_gen.generate(max_degree)
random_loc_surf2 = my_loc_surf_gen.generate(max_degree)
print("local random surface")
print(25*random_loc_surf1(x, y))
print(25*random_loc_surf2(x, y))
my_loc_surf_gen.reset_random()
random_loc_surf3 = my_loc_surf_gen.generate(max_degree)
random_loc_surf4 = my_loc_surf_gen.generate(max_degree)
print("local random surface")
print(25*random_loc_surf3(x, y))
print(25*random_loc_surf4(x, y))