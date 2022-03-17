import numpy as np
import math

inputfile = open("/Users/duanchenda/Desktop/idkresearch/Chenda_Yayun_inverse_glycerin_robot_design/ydu_memory/chenda_inversevalid/forward_lrrate.csv")
f = open('/Users/duanchenda/Desktop/idkresearch/Chenda_Yayun_inverse_glycerin_robot_design/ydu_memory/chenda_inversevalid/Commands.txt','w')
for eachline in inputfile.readlines():
    cmdlist = [float(x) for x in eachline.split(',')]
    print(cmdlist)
    c1 = cmdlist[0]
    c2 = cmdlist[1]
    cr = cmdlist[2]
    rodlen = cmdlist[3]
    headrad = cmdlist[4]
    discrad = cmdlist[5]
    rodrad = cmdlist[6]
    omega = cmdlist[7]
    cmdline = './simDER' + ' option.txt ' + '--' + ' c1 ' +  '%5.5f' % c1 + \
                '--' + ' c2 ' +  '%5.5f' % c2 + '--' + ' cr ' +  '%5.5f' % cr + '--' + \
                    ' RodLength ' +  '%5.5f' % rodlen + '--' + ' headradius ' + '%5.5f' % headrad + '--' + ' discradius ' + '%5.5f' % discrad +\
                        '--' + ' rodRadius ' +  '%5.5f' % rodrad + '--' + ' omega ' +  '%5.5f' % omega + '\n'
    f.write(cmdline)
f.close()

# N1 = 5
# N2 = 5
# N3 = 5
# N4 = 9


# c1 = np.linspace(2,4,N1)
# c2 = np.linspace(1.6,2.4,N2)
# cr = np.linspace(2.45,3.15,N3)

# f = open('Commands.txt','w')
# # c1 = np.linspace(0.1, 4.1, 5)
# # c2 = np.linspace(0.01, 0.05, 5)
# # viscosity = np.linspace(1.0, 4.0, 4)
# # headradius = np.linspace(1.0, 2.5, 4)


# # youngs = np.linspace(0.9e6, 1.2e6, 4)

# # oo = np.linspace(1, 251, 50)
# # # print(oo)
# # omega = [o*2*math.pi/60 for o in oo]
# # print(omega)
# for cc1 in c1:
#     for cc2 in c2:
#         for crr in cr:
#         # for vis in viscosity:
#             # for r in headradius:
#                 # for y in youngs:
#             cmdline = './simDER' + ' option.txt ' + '--' + ' c1 ' +  '%5.5f' % cc1 + '--' + ' c2 ' +  '%5.5f' % cc2 + '--' + ' cr ' +  '%5.5f' % crr + '--' + '\n'
#             f.write(cmdline)
# f.close()


# for ooo in omega:
#     cmdline = './simDER' + ' option.txt ' + '--' + ' omega '+'%5.5f' % ooo + '\n'
#     f.write(cmdline)
# f.close()