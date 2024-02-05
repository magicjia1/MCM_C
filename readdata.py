import pandas as pd
import matplotlib.pyplot as plt
import  numpy as np

def show_time_score(data):
    first_row_index = data.index[0]
    # 创建副本
    copy_data = data.copy()
    id = copy_data['match_id'][first_row_index+0]
    # 创建新列 'elapsed_time_seconds' 存储转换后的值
    copy_data['elapsed_time_seconds'] = pd.to_timedelta(copy_data['elapsed_time'].astype(str)).dt.total_seconds()


    # 画折线图
    plt.plot(copy_data['elapsed_time_seconds'], copy_data['p1_games'], label='Player 1 Games', marker='o')
    plt.plot(copy_data['elapsed_time_seconds'], copy_data['p2_games'], label='Player 2 Games', marker='o')

    # 添加图表标题和标签
    plt.title(f'Player 1 and Player 2 Games Over Time \n  {id}')
    plt.xlabel('Elapsed Time (seconds)')
    plt.ylabel('Number of Games')
    plt.legend()  # 显示图例

    # 显示图表
    plt.show()


def plot_scatter_distance(data):
    # 创建副本
    copy_data = data.copy()

    # 绘制散点图
    plt.scatter(copy_data['p1_distance_run'], copy_data['p2_distance_run'],
                c=copy_data['point_victor'].map({1: 'blue', 2: 'red'}), label='point_winner blue for 1 , red for2',
                alpha=0.5)
    # 画y=x的线
    plt.plot([min(copy_data['p1_distance_run']), max(copy_data['p1_distance_run'])],
             [min(copy_data['p2_distance_run']), max(copy_data['p2_distance_run'])],
             linestyle='--', color='black', label='y=x')
    # 添加图表标题和标签
    plt.title('Scatter Plot of Player 1 vs Player 2 Distance Run')
    plt.xlabel('Player 1 Distance Run')
    plt.ylabel('Player 2 Distance Run')
    plt.legend()  # 显示图例

    # 显示图表
    plt.show()


def plot_rally_count_scatter(data):
    # 创建副本
    copy_data = data.copy()

    # 绘制散点图
    plt.scatter(copy_data['elapsed_time'], copy_data['rally_count'],
                c=copy_data['point_victor'].map({1: 'blue', 2: 'red'}), label='point_winner blue for 1 , red for2',
                alpha=0.5)

    # 添加图表标题和标签
    plt.title('Scatter Plot of Rally Count vs Elapsed Time')
    plt.xlabel('Elapsed Time (seconds)')
    plt.ylabel('Rally Count')
    plt.legend()  # 显示图例

    # 显示图表
    plt.show()


def calculate_matching_elements_ratio(data):
    # 创建 DataFrame
    df = pd.DataFrame(data)

    # 计算数字一样的元素所占比例
    matching_elements_count = (df['server'] == df['point_victor']).sum()
    total_elements = len(df)

    # 计算比例
    matching_elements_ratio = matching_elements_count / total_elements

    return matching_elements_ratio


import momentum


def calculate_accuracy(data):
    df = pd.DataFrame(data)
    copy_data = df.copy()
    list_id = []
    list_momentum1 = []

    list_momentum2 = []
    for index, row in copy_data.iterrows():
        match_score_instance1 = momentum.Momentum(
            player_no=1,
            sets=row['set_no'],
            games=row['game_no'],
            points=row['point_no'],
            server_advantage=row['server'],
            unforced_errors=row['p1_unf_err'],
            double_faults=row['p1_double_fault'],

            consecutive_points=row['consecutive_points'],
            successful_serves=row['p1_ace'],
            successful_returns=row['p1_winner'],
            break_points_successful=row['p1_break_pt_won'],
            ace_situation=row['p1_ace'],

            net_shots=row['p1_net_pt'],
            net_shots_won=row['p1_net_pt_won'],
            consecutive_point_victor_before=row['consecutive_point_victor_before']
        )
        match_score_instance2 = momentum.Momentum(
            player_no=2,
            sets=row['set_no'],
            games=row['game_no'],
            points=row['point_no'],
            server_advantage=row['server'],
            unforced_errors=row['p2_unf_err'],
            double_faults=row['p2_double_fault'],

            consecutive_points=row['consecutive_points'],
            successful_serves=row['p2_ace'],
            successful_returns=row['p2_winner'],
            break_points_successful=row['p2_break_pt_won'],
            ace_situation=row['p2_ace'],
            net_shots=row['p2_net_pt'],
            net_shots_won=row['p2_net_pt_won'],
            consecutive_point_victor_before=row['consecutive_point_victor_before']
        )
        list_id.append(index)
        list_momentum1.append(match_score_instance1.calculate_score())
        list_momentum2.append(match_score_instance2.calculate_score())
    first_row_index = df.index[0]
    id = copy_data['match_id'][0+first_row_index]


    result = [a - b for a, b in zip(list_momentum1, list_momentum2)]




    # 基于result值计算预测
    predictions = ['Player 1' if r >= 0 else 'Player 2' for r in result]

    # 将预测与实际结果进行比较
    actual_outcomes = ['Player 1' if pv == 1 else 'Player 2' for pv in copy_data['point_victor']]

    # 计算准确率
    correct_predictions = sum(1 for p, a in zip(predictions, actual_outcomes) if p == a)
    total_predictions = len(predictions)
    accuracy = correct_predictions / total_predictions * 100

    copy_data['momentum_play1'] = list_momentum1
    copy_data['momentum_play2'] = list_momentum1
    copy_data['momentum_subtract'] = result

    # 选择数值型列
    numeric_columns = copy_data.select_dtypes(include=['float64', 'int64'])

    # 计算相关性
    correlation_matrix = numeric_columns.corr()

    import seaborn as sns
    sns.set_style('whitegrid')

    # # 设置画板尺寸
    # # plt.subplots(figsize=(30, 20))
    #
    # # 画热力图
    # # 为上三角矩阵生成掩码
    # mask = np.zeros_like(numeric_columns.corr(), dtype=bool)
    # mask[np.triu_indices_from(mask)] = True

    # sns.heatmap(numeric_columns.corr(),
    #             cmap=sns.diverging_palette(20, 220, n=200),
    #             mask=mask,  # 数据显示在mask为False的单元格中
    #             annot=True,  # 注入数据
    #             center=0,  # 绘制有色数据时将色彩映射居中的值
    #             )


    # # 设置图像尺寸
    # plt.figure(figsize=(20, 8))
    # sns.set(font_scale=0.7)
    # sns.set_style('whitegrid')
    # # 热力图主要参数调整
    # ax = sns.heatmap(numeric_columns.corr().loc[['momentum_subtract'], :], square=True,
    #                  cmap=sns.diverging_palette(20, 220, n=200),
    #                  annot=True, annot_kws={"size": 8}, vmax=1.0, vmin=-1.0)
    # # 更改坐标轴标签字体大小
    # ax.tick_params(labelsize=7)
    # # 旋转x轴刻度上文字方向
    # # ax.set_xticklabels(ax.get_xticklabels(), rotation=20)
    #
    # # Give title.
    # plt.title("Heatmap of all the Features", fontsize=30)
    # plt.show()



    # 提取 'momentum_subtract' 列与其他列的相关性
    momentum_subtract_correlation = correlation_matrix['momentum_subtract'].drop('momentum_subtract')


    # momentum_subtract_correlation = correlation_matrix['point2'].drop('point2')
    # momentum_subtract_correlation = correlation_matrix['point1'].drop('point1')

    # # 打印 'momentum_subtract' 列与其他列的相关性
    # print("momentum_subtract 与其他列的相关性:")
    # print(momentum_subtract_correlation)
    # print(type(momentum_subtract_correlation))

    # 按相关性值排序
    # momentum_subtract_correlation_sorted = momentum_subtract_correlation.sort_values(ascending=False)
    #
    # # 打印排序后的相关性
    # print("排序后的相关性:",id)
    # print(momentum_subtract_correlation_sorted)

    return accuracy, momentum_subtract_correlation


def show_momentum(data):
    first_row_index = data.index[0]
    df = pd.DataFrame(data)
    copy_data = df.copy()
    list_id = []
    list_momentum1 = []

    list_momentum2 = []
    for index, row in copy_data.iterrows():
        match_score_instance1 = momentum.Momentum(
            player_no=1,
            sets=row['set_no'],
            games=row['game_no'],
            points=row['point_no'],
            server_advantage=row['server'],
            unforced_errors=row['p1_unf_err'],
            double_faults=row['p1_double_fault'],

            consecutive_points=row['consecutive_points'],
            successful_serves=row['p1_ace'],
            successful_returns=row['p1_winner'],
            break_points_successful=row['p1_break_pt_won'],
            ace_situation=row['p1_ace'],
            net_shots=row['p1_net_pt'],
            net_shots_won=row['p1_net_pt_won'],
            consecutive_point_victor_before=row['consecutive_point_victor_before']
        )
        match_score_instance2 = momentum.Momentum(
            player_no=2,
            sets=row['set_no'],
            games=row['game_no'],
            points=row['point_no'],
            server_advantage=row['server'],
            unforced_errors=row['p2_unf_err'],
            double_faults=row['p2_double_fault'],

            consecutive_points=row['consecutive_points'],
            successful_serves=row['p2_ace'],
            successful_returns=row['p2_winner'],
            break_points_successful=row['p2_break_pt_won'],
            ace_situation=row['p2_ace'],
            net_shots=row['p2_net_pt'],
            net_shots_won=row['p1_net_pt_won'],
            consecutive_point_victor_before=row['consecutive_point_victor_before']
        )
        list_id.append(index)
        list_momentum1.append(match_score_instance1.calculate_score())
        list_momentum2.append(match_score_instance2.calculate_score())

    id = copy_data['match_id'][first_row_index+0]
    # 创建新列 'elapsed_time_seconds' 存储转换后的值
    # copy_data['elapsed_time_seconds'] = pd.to_timedelta(copy_data['elapsed_time'].astype(str)).dt.total_seconds()

    result = [a - b for a, b in zip(list_momentum1, list_momentum2)]

    # # 画折线图
    # plt.scatter(list_id, list_momentum1, label='Player 1 momentum', marker='o')
    # plt.scatter(list_id, list_momentum2, label='Player 2 momentum', marker='o')
    #
    # # 添加图表标题和标签
    # plt.title('Player 1 and Player 2 Games Over Time')
    # plt.xlabel('Elapsed Time (seconds)')
    # plt.ylabel('Momentum of Games')
    #
    # # 显示图例
    # plt.legend()
    #
    # # 显示图表
    # plt.show()

    # 遍历数据，根据 point_victor 的值设置颜色
    # 定义两个点
    # 创建一个更大的图表

    plt.figure(figsize=(10, 6))

    # # 创建第一个图形
    # plt.figure(1)
    # # 绘制第一个子图
    # plt.subplot(311)  # 表示3行1列的第1个子图
    # # 遍历数据并画散点图
    # for x, y, point_victor in zip(list_id, list_momentum1, copy_data['point_victor']):
    #     color = 'blue' if point_victor == 1 else 'red'
    #     plt.scatter(x, y, color=color)
    #
    # plt.title(f'Momentum player1 {id}')
    #
    # # 创建第二个图形
    # plt.figure(2)
    #
    # # 绘制第二个子图
    # plt.subplot(211)  # 表示2行1列的第1个子图
    # # 遍历数据并画散点图
    # for x, y, point_victor in zip(list_id, list_momentum2, copy_data['point_victor']):
    #     color = 'blue' if point_victor == 1 else 'red'
    #     plt.scatter(x, y, color=color)
    #
    # plt.title(f'Momentum player2 {id}')

    # 创建第三个图形
    # plt.figure(1)

    # 绘制第三个子图
    # plt.subplot(111)  # 表示1行1列的第1个子图

    # 定义两个点
    blue_point = plt.scatter([], [], color='blue', label='Player 1 (point_victor = 1)')
    red_point = plt.scatter([], [], color='red', label='Player 2 (point_victor = 2)')

    # 遍历数据并画散点图
    for x, y, point_victor in zip(list_id, result, copy_data['point_victor']):
        color = 'blue' if point_victor == 1 else 'red'
        plt.scatter(x, y, color=color)

    # 基于result值计算预测
    predictions = ['Player 1' if r > 0 else 'Player 2' for r in result]

    # 将预测与实际结果进行比较
    actual_outcomes = ['Player 1' if pv == 1 else 'Player 2' for pv in copy_data['point_victor']]

    # 计算准确率
    correct_predictions = sum(1 for p, a in zip(predictions, actual_outcomes) if p == a)
    total_predictions = len(predictions)
    accuracy = correct_predictions / total_predictions * 100

    # 在图表底部显示准确率信息
    plt.text(0.5, -1, f'准确率: {accuracy:.2f}%', transform=plt.gca().transAxes, ha='center', color='green')

    # 添加图表标题和标签
    plt.title(f'Momentum: Player 1 - Player 2,{id}')
    plt.xlabel('Elapsed Time (line)')
    plt.ylabel('Momentum of Subtract')

    # 显示图例
    plt.legend(handles=[blue_point, red_point], title='Color Legend:' ,loc='upper right')

    # 显示图表
    plt.show()


def add_consecutive_wins(df):
    first_row_index = df.index[0]
    # 新增列记录连胜次数
    df.loc[:, 'consecutive_points0'] = 0
    df.loc[:, 'consecutive_point_victor_before'] = 0
    df.loc[:, 'consecutive_points'] = 0

    # 循环遍历DataFrame 找出连胜次数
    current_streak = 0
    for index, row in df.iterrows():
        if index > first_row_index+0 and row['point_victor'] == df.at[index - 1, 'point_victor']:
            current_streak += 1
        else:
            current_streak = 1
        df.at[index, 'consecutive_points0'] = current_streak

        # 追踪上一把的获胜者
        if index == first_row_index+0 :
            df.at[index, 'consecutive_point_victor_before'] = 0
        if index >  first_row_index+0 :
            df.at[index, 'consecutive_point_victor_before'] = df.at[index - 1, 'point_victor']

    for index, row in df.iterrows():
        if index ==  first_row_index+0 :
            df.at[index, 'consecutive_points'] = 0
        if index > first_row_index+0 :
            df.at[index, 'consecutive_points'] = df.at[index - 1, 'consecutive_points0']

    return df



if __name__ == '__main__':

    # csv_file_path = 'Wimbledon_featured_matches.xlsx'
    # df = pd.read_csv(csv_file_path,encoding='utf-8')
    # print(df)
    # # 按照 'match_id' 分组
    # grouped_data = df.groupby('match_id')
    # print( grouped_data.groups)

    # 读取Excel文件
    # csv_file_path = '../Wimbledon_featured_matches.xlsx'
    csv_file_path ='2023-wimbledon-女单(5场).xlsx'
    df = pd.read_excel(csv_file_path, )


    # 按照 'match_id' 分组
    grouped_data = df.groupby('match_id')

    # 打印分组的数量
    print("分组数量：", len(grouped_data))
    # 打印每个组的数据
    for group_name, group_data in grouped_data:
        print(f"Group {group_name} has {len(group_data)} rows.")


    group_keys = list(grouped_data.groups.keys())

    ## 单个id的比赛的结果
    second_group = grouped_data.get_group(group_keys[1])
    pd.set_option('display.width', 5000)
    pd.set_option('display.max_columns', 5)
    pd.set_option('display.max_rows', None)
    # 打印列名
    print(second_group.columns)
    # show_time_score(second_group)
    second_group = add_consecutive_wins(second_group)
    # show_momentum(second_group)
    # calculate_accuracy(second_group)




    res ={}
    # 存储多个 'momentum_subtract_correlation'
    momentum_subtract_correlations = []

    # 循环迭代
    for group_key in group_keys:
        group = grouped_data.get_group(group_key)
        group = add_consecutive_wins(group)

        # 调用 calculate_accuracy 函数
        acc, momentum_subtract_correlation = calculate_accuracy(group)
        momentum_subtract_correlations.append(momentum_subtract_correlation)
        res[group_key]=acc
    print(res)


    # # 将 Series 转换为 DataFrame
    #
    # df = pd.DataFrame(momentum_subtract_correlations)
    #
    # # 计算每个元素的平均值
    # average_values = df.mean()
    #
    # # 打印平均值
    # print("每个元素的平均值:")
    # print(average_values)
    #
    # # 按相关性值排序
    # average_values_sorted = average_values.sort_values(ascending=False)
    #
    # # 打印排序后的相关性
    # print("排序后的相关性:")
    # print(average_values_sorted)
    # # print(res)



    # plot_scatter_distance(first_group_data)
    # plot_rally_count_scatter(first_group_data)
    #
    #
    #
    # a= calculate_matching_elements_ratio(first_group_data)
    # print( a)

    # show_momentum(first_group_data)

    # # 遍历分组并打印每个分组的数据

    # for group_name, group_data in grouped_data:
    #     copy_data = group_data.copy()
    #     copy_data = add_consecutive_wins(copy_data)
    #     show_momentum(copy_data)
    #     print(calculate_matching_elements_ratio(group_data))
