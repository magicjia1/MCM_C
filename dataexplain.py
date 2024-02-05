import numpy as np

baseline_data_values = [
    0.65, 0.7611940298507462, 0.6865671641791045, 0.6617210682492581,
    0.6585365853658537, 0.7078313253012049, 0.7413793103448276,
    0.7263157894736842, 0.6666666666666666, 0.5754716981132075,
    0.6529411764705882, 0.5781818181818181, 0.6275862068965518,
    0.6756756756756757, 0.6919191919191919, 0.688622754491018,
    0.6622222222222223, 0.7043795620437956, 0.6311475409836066,
    0.6958041958041958, 0.6744186046511628, 0.6666666666666666,
    0.7341389728096677, 0.7601476014760148, 0.708994708994709,
    0.6819787985865724, 0.6580310880829016, 0.663594470046083,
    0.610062893081761, 0.6831683168316832, 0.6107784431137725
]
baseline = {
    '2023-wimbledon-1301': 0.65, '2023-wimbledon-1302': 0.7611940298507462,
    '2023-wimbledon-1303': 0.6865671641791045, '2023-wimbledon-1304': 0.6617210682492581,
    '2023-wimbledon-1305': 0.6585365853658537, '2023-wimbledon-1306': 0.7078313253012049,
    '2023-wimbledon-1307': 0.7413793103448276, '2023-wimbledon-1308': 0.7263157894736842,
    '2023-wimbledon-1309': 0.6666666666666666, '2023-wimbledon-1310': 0.5754716981132075,
    '2023-wimbledon-1311': 0.6529411764705882, '2023-wimbledon-1312': 0.5781818181818181,
    '2023-wimbledon-1313': 0.6275862068965518, '2023-wimbledon-1314': 0.6756756756756757,
    '2023-wimbledon-1315': 0.6919191919191919, '2023-wimbledon-1316': 0.688622754491018,
    '2023-wimbledon-1401': 0.6622222222222223, '2023-wimbledon-1402': 0.7043795620437956,
    '2023-wimbledon-1403': 0.6311475409836066, '2023-wimbledon-1404': 0.6958041958041958,
    '2023-wimbledon-1405': 0.6744186046511628, '2023-wimbledon-1406': 0.6666666666666666,
    '2023-wimbledon-1407': 0.7341389728096677, '2023-wimbledon-1408': 0.7601476014760148,
    '2023-wimbledon-1501': 0.708994708994709, '2023-wimbledon-1502': 0.6819787985865724,
    '2023-wimbledon-1503': 0.6580310880829016, '2023-wimbledon-1504': 0.663594470046083,
    '2023-wimbledon-1601': 0.610062893081761, '2023-wimbledon-1602': 0.6831683168316832,
    '2023-wimbledon-1701': 0.6107784431137725
}
# 将字典中的每个值乘以100
baseline_times_100 = {key: value * 100 for key, value in baseline.items()}

# average_value = np.mean(baseline_data_values)
# print(f"The average value is: {average_value}")


# origin datares
# datares ={'2023-wimbledon-1301': 81.33333333333333, '2023-wimbledon-1302': 81.09452736318407, '2023-wimbledon-1303': 77.61194029850746, '2023-wimbledon-1304': 83.38278931750742, '2023-wimbledon-1305': 81.30081300813008, '2023-wimbledon-1306': 79.81927710843374, '2023-wimbledon-1307': 82.32758620689656, '2023-wimbledon-1308': 77.36842105263158, '2023-wimbledon-1309': 83.09859154929578, '2023-wimbledon-1310': 82.38993710691824, '2023-wimbledon-1311': 83.52941176470588, '2023-wimbledon-1312': 84.0, '2023-wimbledon-1313': 77.93103448275862, '2023-wimbledon-1314': 84.86486486486487, '2023-wimbledon-1315': 81.81818181818183, '2023-wimbledon-1316': 82.03592814371258, '2023-wimbledon-1401': 80.88888888888889, '2023-wimbledon-1402': 79.1970802919708, '2023-wimbledon-1403': 85.24590163934425, '2023-wimbledon-1404': 81.81818181818183, '2023-wimbledon-1405': 79.53488372093022, '2023-wimbledon-1406': 79.48717948717949, '2023-wimbledon-1407': 82.17522658610272, '2023-wimbledon-1408': 80.81180811808119, '2023-wimbledon-1501': 80.42328042328042, '2023-wimbledon-1502': 82.68551236749117, '2023-wimbledon-1503': 80.31088082901555, '2023-wimbledon-1504': 79.26267281105991, '2023-wimbledon-1601': 78.61635220125787, '2023-wimbledon-1602': 80.6930693069307, '2023-wimbledon-1701': 77.24550898203593}


# only no consecutive_points
# datares = {'2023-wimbledon-1301': 90.33333333333333, '2023-wimbledon-1302': 95.02487562189054, '2023-wimbledon-1303': 94.02985074626866, '2023-wimbledon-1304': 94.3620178041543, '2023-wimbledon-1305': 94.3089430894309, '2023-wimbledon-1306': 93.07228915662651, '2023-wimbledon-1307': 94.82758620689656, '2023-wimbledon-1308': 91.57894736842105, '2023-wimbledon-1309': 94.83568075117371, '2023-wimbledon-1310': 92.13836477987421, '2023-wimbledon-1311': 92.94117647058823, '2023-wimbledon-1312': 94.54545454545455, '2023-wimbledon-1313': 88.96551724137932, '2023-wimbledon-1314': 97.2972972972973, '2023-wimbledon-1315': 94.94949494949495, '2023-wimbledon-1316': 92.81437125748504, '2023-wimbledon-1401': 93.33333333333333, '2023-wimbledon-1402': 91.6058394160584, '2023-wimbledon-1403': 95.08196721311475, '2023-wimbledon-1404': 94.4055944055944, '2023-wimbledon-1405': 90.69767441860465, '2023-wimbledon-1406': 94.87179487179486, '2023-wimbledon-1407': 93.35347432024169, '2023-wimbledon-1408': 94.8339483394834, '2023-wimbledon-1501': 93.65079365079364, '2023-wimbledon-1502': 95.40636042402826, '2023-wimbledon-1503': 91.70984455958549, '2023-wimbledon-1504': 88.47926267281106, '2023-wimbledon-1601': 88.67924528301887, '2023-wimbledon-1602': 92.57425742574257, '2023-wimbledon-1701': 90.41916167664671}

# only no server advantage
# datares = {'2023-wimbledon-1301': 74.0, '2023-wimbledon-1302': 72.636815920398, '2023-wimbledon-1303': 73.88059701492537, '2023-wimbledon-1304': 79.22848664688428, '2023-wimbledon-1305': 79.26829268292683, '2023-wimbledon-1306': 72.89156626506023, '2023-wimbledon-1307': 77.15517241379311, '2023-wimbledon-1308': 67.36842105263158, '2023-wimbledon-1309': 79.81220657276995, '2023-wimbledon-1310': 78.61635220125787, '2023-wimbledon-1311': 78.23529411764706, '2023-wimbledon-1312': 78.18181818181819, '2023-wimbledon-1313': 73.79310344827587, '2023-wimbledon-1314': 78.91891891891892, '2023-wimbledon-1315': 77.27272727272727, '2023-wimbledon-1316': 78.44311377245509, '2023-wimbledon-1401': 76.0, '2023-wimbledon-1402': 72.26277372262774, '2023-wimbledon-1403': 79.50819672131148, '2023-wimbledon-1404': 73.77622377622379, '2023-wimbledon-1405': 74.4186046511628, '2023-wimbledon-1406': 77.43589743589745, '2023-wimbledon-1407': 75.22658610271903, '2023-wimbledon-1408': 76.38376383763837, '2023-wimbledon-1501': 72.4867724867725, '2023-wimbledon-1502': 77.03180212014135, '2023-wimbledon-1503': 76.68393782383419, '2023-wimbledon-1504': 71.42857142857143, '2023-wimbledon-1601': 79.24528301886792, '2023-wimbledon-1602': 74.25742574257426, '2023-wimbledon-1701': 71.55688622754491}

 #only no unforced_errors
datares = {'2023-wimbledon-1301': 67.66666666666666, '2023-wimbledon-1302': 74.12935323383084,
           '2023-wimbledon-1303': 72.38805970149254, '2023-wimbledon-1304': 73.88724035608308,
           '2023-wimbledon-1305': 67.88617886178862, '2023-wimbledon-1306': 71.6867469879518,
           '2023-wimbledon-1307': 78.44827586206897, '2023-wimbledon-1308': 68.42105263157895,
           '2023-wimbledon-1309': 73.23943661971832, '2023-wimbledon-1310': 66.98113207547169,
           '2023-wimbledon-1311': 75.88235294117646, '2023-wimbledon-1312': 66.9090909090909,
           '2023-wimbledon-1313': 69.3103448275862, '2023-wimbledon-1314': 71.35135135135135,
           '2023-wimbledon-1315': 74.74747474747475, '2023-wimbledon-1316': 68.8622754491018,
           '2023-wimbledon-1401': 69.77777777777779, '2023-wimbledon-1402': 68.97810218978103,
           '2023-wimbledon-1403': 73.77049180327869, '2023-wimbledon-1404': 73.07692307692307,
           '2023-wimbledon-1405': 69.30232558139535, '2023-wimbledon-1406': 71.28205128205128,
           '2023-wimbledon-1407': 74.92447129909365, '2023-wimbledon-1408': 75.64575645756457,
           '2023-wimbledon-1501': 73.01587301587301, '2023-wimbledon-1502': 73.4982332155477,
           '2023-wimbledon-1503': 74.61139896373057, '2023-wimbledon-1504': 70.96774193548387,
           '2023-wimbledon-1601': 72.95597484276729, '2023-wimbledon-1602': 71.28712871287128,
           '2023-wimbledon-1701': 67.66467065868264}

#only no double_faults
# datares = {'2023-wimbledon-1301': 81.0, '2023-wimbledon-1302': 81.09452736318407, '2023-wimbledon-1303': 77.61194029850746, '2023-wimbledon-1304': 82.7893175074184, '2023-wimbledon-1305': 80.89430894308943, '2023-wimbledon-1306': 79.81927710843374, '2023-wimbledon-1307': 82.32758620689656, '2023-wimbledon-1308': 76.84210526315789, '2023-wimbledon-1309': 82.15962441314554, '2023-wimbledon-1310': 82.38993710691824, '2023-wimbledon-1311': 82.94117647058825, '2023-wimbledon-1312': 83.63636363636363, '2023-wimbledon-1313': 77.93103448275862, '2023-wimbledon-1314': 84.32432432432432, '2023-wimbledon-1315': 81.31313131313132, '2023-wimbledon-1316': 81.437125748503, '2023-wimbledon-1401': 80.44444444444444, '2023-wimbledon-1402': 78.46715328467153, '2023-wimbledon-1403': 85.24590163934425, '2023-wimbledon-1404': 81.46853146853147, '2023-wimbledon-1405': 79.53488372093022, '2023-wimbledon-1406': 79.48717948717949, '2023-wimbledon-1407': 81.57099697885197, '2023-wimbledon-1408': 80.44280442804428, '2023-wimbledon-1501': 79.8941798941799, '2023-wimbledon-1502': 82.3321554770318, '2023-wimbledon-1503': 79.79274611398964, '2023-wimbledon-1504': 79.26267281105991, '2023-wimbledon-1601': 78.61635220125787, '2023-wimbledon-1602': 80.6930693069307, '2023-wimbledon-1701': 77.24550898203593}



#   no unforced_errors and no consecutive_points
# datares = {'2023-wimbledon-1301': 77.0, '2023-wimbledon-1302': 83.5820895522388, '2023-wimbledon-1303': 82.83582089552239, '2023-wimbledon-1304': 80.41543026706232, '2023-wimbledon-1305': 79.67479674796748, '2023-wimbledon-1306': 82.53012048192771, '2023-wimbledon-1307': 87.93103448275862, '2023-wimbledon-1308': 81.57894736842105, '2023-wimbledon-1309': 84.97652582159625, '2023-wimbledon-1310': 75.47169811320755, '2023-wimbledon-1311': 82.35294117647058, '2023-wimbledon-1312': 76.36363636363637, '2023-wimbledon-1313': 78.62068965517241, '2023-wimbledon-1314': 81.08108108108108, '2023-wimbledon-1315': 87.87878787878788, '2023-wimbledon-1316': 78.44311377245509, '2023-wimbledon-1401': 78.66666666666666, '2023-wimbledon-1402': 82.84671532846716, '2023-wimbledon-1403': 72.95081967213115, '2023-wimbledon-1404': 82.51748251748252, '2023-wimbledon-1405': 80.93023255813954, '2023-wimbledon-1406': 83.58974358974359, '2023-wimbledon-1407': 86.70694864048339, '2023-wimbledon-1408': 85.60885608856088, '2023-wimbledon-1501': 86.24338624338624, '2023-wimbledon-1502': 84.09893992932862, '2023-wimbledon-1503': 82.90155440414507, '2023-wimbledon-1504': 78.3410138248848, '2023-wimbledon-1601': 80.50314465408806, '2023-wimbledon-1602': 81.68316831683168, '2023-wimbledon-1701': 78.74251497005989}


# # no ace
# datares={'2023-wimbledon-1301': 80.66666666666666, '2023-wimbledon-1302': 79.60199004975125, '2023-wimbledon-1303': 75.3731343283582, '2023-wimbledon-1304': 82.19584569732937, '2023-wimbledon-1305': 81.30081300813008, '2023-wimbledon-1306': 78.91566265060241, '2023-wimbledon-1307': 81.46551724137932, '2023-wimbledon-1308': 76.84210526315789, '2023-wimbledon-1309': 81.69014084507043, '2023-wimbledon-1310': 81.13207547169812, '2023-wimbledon-1311': 82.35294117647058, '2023-wimbledon-1312': 82.54545454545455, '2023-wimbledon-1313': 76.89655172413794, '2023-wimbledon-1314': 84.32432432432432, '2023-wimbledon-1315': 79.7979797979798, '2023-wimbledon-1316': 82.03592814371258, '2023-wimbledon-1401': 80.44444444444444, '2023-wimbledon-1402': 78.83211678832117, '2023-wimbledon-1403': 84.42622950819673, '2023-wimbledon-1404': 81.46853146853147, '2023-wimbledon-1405': 78.13953488372093, '2023-wimbledon-1406': 79.48717948717949, '2023-wimbledon-1407': 80.06042296072508, '2023-wimbledon-1408': 78.59778597785979, '2023-wimbledon-1501': 78.83597883597884, '2023-wimbledon-1502': 80.21201413427562, '2023-wimbledon-1503': 78.75647668393782, '2023-wimbledon-1504': 78.80184331797236, '2023-wimbledon-1601': 78.61635220125787, '2023-wimbledon-1602': 80.19801980198021, '2023-wimbledon-1701': 77.24550898203593}

# female people
datares ={'2023-wimbledon-2503': 86.26373626373626, '2023-wimbledon-2504': 83.96946564885496, '2023-wimbledon-2601': 92.62295081967213, '2023-wimbledon-2602': 92.05607476635514, '2023-wimbledon-2701': 87.2}

for k, v in datares.items():
    print(k, v)
# 计算平均值
average_value = sum(datares.values()) / len(datares)

print("平均值:", average_value)


import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


#
def showacc(data):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    import numpy as np

    datares = data
    # Normalize the values to have a maximum of 100
    max_value = 100
    normalized_values = {k: v / max_value * 100 for k, v in datares.items()}

    # Define your own custom normalization range
    custom_norm = Normalize(vmin=60, vmax=100)

    # Use 'viridis' colormap for color variation, apply custom normalization
    colors = cm.viridis(custom_norm(np.array(list(normalized_values.values()))))

    # Plotting
    plt.figure(figsize=(10, 6))
    bars = plt.bar(normalized_values.keys(), normalized_values.values(), color=colors)
    aa = "Only serve advantage"
    bb = "Momentum"
    plt.title(f'{bb} predict point accuracy for every match (Max: 100)')
    plt.xlabel('Matches')
    plt.ylabel('point accuracy')
    plt.xticks(rotation=45, ha='right')

    # Set y-axis limit to 100
    plt.ylim(0, 100)

    # Add colorbar for reference
    mappable = cm.ScalarMappable(cmap=cm.viridis, norm=custom_norm)
    mappable.set_array(list(normalized_values.values()))
    cbar = plt.colorbar(mappable, orientation='vertical')
    cbar.set_label('Normalized Values', rotation=270, labelpad=15)

    plt.tight_layout()
    plt.show()


# 使用示例
# 假设有一个名为 data 的字典
# showacc(data)


# 使用示例
# 假设有一个名为 data 的字典
showacc(datares)