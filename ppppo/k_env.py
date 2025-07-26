import math
import random
import numpy as np
import torch


class MECEnv(object):
    I = 4  # IoT设备的数量
    K = 4  # 无人机UAV的数量   （SBS与MEC服务器是一个整体，两者可以互通）
    M = 1  # MBS的数量
    T = 1000  # 完成整个过程需要的时间   （时间范围划分为t个时隙）
    delta_t = 20  # 时隙的长度
    resolution_options = [
        (256, 144),  # 144P
        (426, 240),  # 240P
        (640, 360),  # 360P
        (854, 480),  # 480P
        (1280, 720),  # 720P
        (1920, 1080)  # 1080P
    ]
    D_i_list = np.random.randint(5, 8, I) * 10 ** 5  # 输入任务的数据大小，单位为Mb(状态1)
    C_i_list = np.zeros(I)  # 任务处理工作量，单位为M CPU cycles（状态2）,目前当作计算资源的大小
    T_i_max = delta_t  # 允许最大延迟
    r_i_k = np.random.randint(5, 10, I) * 10 ** 6  # UE和SBS之间的链路速率,单位Mbps  （状态4）
    f_i_l = np.random.uniform(0.5, 1) * 10 ** 6  # 用户本地处理能力，单位CPU cycles/s
    a_i_k = np.random.randint(0, 2, size=[I, K])  # UE i 与SBS k 之间链路是否可用，可用为1，不可为0.这里为随机取值（状态3）
    # A_link = a_i_k
    f_s = 1 * 10 ** 9  # 每个计算资源块的处理能力（CPU cycles/s）
    # r_m = np.random.randint(1, 3, I) * 10 ** 6  # MBS为每个UE i分配相同数量的带宽rm（t）,单位Mbps （状态6）
    # f_m = np.random.randint(60, 61) * 10 ** 9  # MBS分配UE的计算资源（计算功率），单位CPU cycles/s （状态7）
    U_k = np.random.randint(20, 31, K)  # 每个SBS的计算资源块的数量 （状态5）
    W_k_c = random.uniform(0, 1)  # 时隙节点的信用权重,节点的可信度,每个项的值为[0,1]
    W_k_star = random.uniform(0, 1)
    W_k = random.uniform(0, 1)  # 时隙节点的信用权重,节点的可信度,每个项的值为[0,1]  （状态8）
    s_k = np.random.randint(0, 2, K)  # 是否是恶意节点的标志 （状态9）
    # print("s_k1=",s_k)
    k = np.random.randint(100, 1000)  # 共识节点的个数
    N_fail = np.random.randint(0, 2)  # 时隙失败的次数（状态10）
    omega_k_c = np.random.randint(0, 2)
    omega_k_star = np.random.randint(0, 2)  # 主节点是恶意节点时是否作恶 （状态12）
    omega_k = np.random.randint(0, 2)  # 从节点是恶意节点时是否作恶（状态13）
    T_1_1_mali = random.uniform(0, 2)
    T_2_1_mali = random.uniform(0, 2)  # 主节点是普通节点时作恶的时延 （状态14）
    T_4_k_mali = random.uniform(0.2, 0.8)  # 从节点是普通节点时作恶的时延（状态15）
    K_c_mali = np.random.randint(0, 100, K)  # 所有节点中恶意节点的个数 （状态16）
    T_block_ave = np.random.randint(0, 2)  # 平均共识时延,应该根据公式26计算得到，这里占位（状态11）

    theta = np.random.randint(50, 55) * 10 ** 6  # 生成/验证一个签名的工作量。单位是cycles/s
    sigma = np.random.randint(50, 55) * 10 ** 6  # 生成/验证一个MAC的工作量。单位是cycles/s
    U_k_star = U_k  # 计算资源总数  两者大小相同，只不过一个是主节点中的带*，一个是非主节点中的为U_k
    # K_c = np.random.choice([0, 1], 2 * K)  # 委员会的集合（公式10）
    # k_c = 4
    omega_c = random.uniform(0, 1)
    omega_r = random.uniform(0, 1 - omega_c)
    omega_d = 1 - omega_c - omega_r
    # print("omega_c=",omega_c,"omega_r=",omega_r,"omega_d=",omega_d)
    Thre_h_w = 1.0  # 共识过程中信用值上限
    delta_cli = 0.1  # 对客户端进行延迟惩罚因子
    delta_pri = 0.1  # 对主节点进行延迟惩罚因子
    delta_con = 0.2
    lambda1 = 0.5  # 公式22中的调节因子，为了保持延迟和权重在相同量级的参数
    task_density = 10
    chi = 8
    xi = 3
    # action_bound = [-1,1] # 对应tahn激活函数
    action_bound = 1.0
    # action_dim = I + I + K + K  # 1.用户任务卸载 2.任务处理中计算资源块的分配 3.共识主节点的选择 4.共识非主节点的选择  16dim
    action_dim = I + I + I + K + K + K + K  # 1.用户任务卸载 2.像素的选择 3.任务处理中计算资源块的分配 4.传输速率分配 5.客户端的选择 6共识主节点的选择  28dim。
    # state_dim = I + I + I * K + I + K + I + 1 + 1 + K + I + 1 + 1 + 1 + 1 + 1 + K  # 55dim
    state_dim = I * 4 + K * 4 + I * K + 11

    # alpha2 = 1
    # 1.输入任务的数据大小 I
    # 2.任务处理工作量 I
    # 3.IoT设备和UAV之间的是否链接 I * K
    # 4.IoT设备和UAV之间链路速率 I
    # 5.每个UAV的计算资源块数量  K
    # 6.IoT设备和MBS之间的链路速率 I
    # 7.MBS分配给每个任务的计算资源  1
    # 8.最后一个时隙节点的信用权重   1
    # 9.是否恶意节点的标志   K * 2
    # 10.上一时隙失败的次数   I
    # 11.平均共识延迟  1
    # 12.主节点是恶意节点时是否作恶  1
    # 13.从节点是恶意节点时是否作恶  1
    # 14.主节点是普通节点时作恶的延迟 1*1
    # 15.从节点是普通节点时作恶的延迟 1*1
    # 16.所有节点中恶意节点的个数    K

    # alpha2 = 1
    # 1.输入任务的数据大小 I
    # 2.任务处理工作量 I
    # 4.IoT设备和UAV之间链路速率 I
    # 5.每个UAV的计算资源块数量  K
    # 8.最后一个时隙节点的信用权重   1*3
    # 9.是否恶意节点的标志   K * 2
    # 10.上一时隙失败的次数   I
    # 11.平均共识延迟  1
    # 12.主节点是恶意节点时是否作恶  1*3
    # 14.主节点是普通节点时作恶的延迟 1*3
    # 16.所有节点中恶意节点的个数    K
    # 3.IoT设备和UAV之间的是否链接 I * K

    def __init__(self):  # 在类中定义函数，每个类中第一个必须定义_int_函数。每个定义的函数的第一个参数必须是self，调用时不用传递self参数。该句以冒号结束
        # 需要一个总的集合,把所有状态放入该集合中
        self.start_state = np.append(self.D_i_list, self.C_i_list)  # D_i_list输入任务的数据大小，C_i_list任务工作量
        self.start_state = np.append(self.start_state, self.r_i_k)  # IoT设备和UAV之间链路速率
        self.start_state = np.append(self.start_state, self.U_k)  # 每个UAV的计算资源块数量
        # self.start_state = np.append(self.start_state, self.r_m)  # IoT设备和MBS之间的链路速率
        # self.start_state = np.append(self.start_state, self.f_m)  # MBS分配给每个任务的计算资源
        self.start_state = np.append(self.start_state, self.W_k_c)
        self.start_state = np.append(self.start_state, self.W_k_star)
        self.start_state = np.append(self.start_state, self.W_k)  # 时隙节点的信用权重
        self.start_state = np.append(self.start_state, self.s_k)  # 是否是恶意节点的标志
        self.start_state = np.append(self.start_state, self.N_fail)  # 时隙失败的次数
        self.start_state = np.append(self.start_state, self.T_block_ave)  # 平均共识时延
        self.start_state = np.append(self.start_state, self.omega_k_c)  # 客户端是恶意节点时是否作恶
        self.start_state = np.append(self.start_state, self.omega_k_star)  # 主节点是恶意节点时是否作恶
        self.start_state = np.append(self.start_state, self.omega_k)  # 从节点是恶意节点时是否作恶
        self.start_state = np.append(self.start_state, self.T_1_1_mali)  # 客户端的恶意延迟
        self.start_state = np.append(self.start_state, self.T_2_1_mali)  # 主节点的恶意时延
        self.start_state = np.append(self.start_state, self.T_4_k_mali)  # 从节点作恶的时延
        self.start_state = np.append(self.start_state, self.K_c_mali)  # 所有节点中恶意节点的个数
        self.start_state = np.append(self.start_state, np.ravel(self.a_i_k))
        self.state = self.start_state

    def reset_env(self):
        self.D_i_list = np.random.randint(5, 10, self.I) * 10 ** 5
        self.C_i_list = np.random.randint(10, 20, self.I) * 10 ** 5
        self.r_i_k = np.random.randint(5, 10, self.I) * 10 ** 6
        self.U_k = np.random.randint(20, 51, self.K)
        # self.r_m = np.random.randint(1, 5, self.I) * 10 ** 6
        # self.f_m = np.random.randint(50, 101) * 10 ** 9
        self.W_k_c = random.uniform(0, 1)
        self.W_k_star = random.uniform(0, 1)
        self.W_k = random.uniform(0, 1)
        self.s_k = np.random.randint(0, 2, self.K)
        # print("s_k2=", self.s_k)
        self.N_fail = np.random.randint(0, 2)
        self.T_block_ave = np.random.randint(0, 2)
        self.omega_k_star = np.random.randint(0, 2)
        self.omega_k = np.random.randint(0, 2)
        self.T_1_1_mali = random.uniform(0, 2)
        self.T_2_1_mali = random.uniform(0, 2)
        self.T_4_k_mali = random.uniform(0, 11)
        self.K_c_mali = np.random.randint(0, 100, self.K)
        self.reset_step()

    # 这里要写的是，这几个动作不仅要在reset_step更新，还要在step中更新。
    # 相当于状态转移，有公式在后面用公式，没有公式用这个随机给

    def reset_step(self):
        # self.a_i_k = np.random.randint(1, 2, size=[self.I, self.K])
        # for i in range(self.I):
        #     b = np.random.randint(0, self.K)
        #     self.a_i_k[i][b] = 0.76
        self.a_i_k = np.random.choice([0, 1], size=[self.I, self.K], p=[0.24, 0.76])

    def reset(self):
        self.reset_step()
        self.state = np.append(self.D_i_list, self.C_i_list)
        self.state = np.append(self.state, self.r_i_k)  # UE和SBS之间的链路速率
        self.state = np.append(self.state, self.U_k)  # 每个SBS的计算资源块的数量
        # self.state = np.append(self.state, self.r_m)  # UE和MBS之间的传输速率
        # self.state = np.append(self.state, self.f_m)  # MBS分配给每个任务的计算资源
        self.state = np.append(self.state, self.W_k_c)
        self.state = np.append(self.state, self.W_k_star)
        self.state = np.append(self.state, self.W_k)  # 时隙节点的信用权重
        self.state = np.append(self.state, self.s_k)  # 是否是恶意节点的标志
        self.state = np.append(self.state, self.N_fail)  # 时隙失败的次数
        self.state = np.append(self.state, self.T_block_ave)  # 平均共识时延
        self.state = np.append(self.state, self.omega_k_star)  # 主节点是恶意节点时是否作恶
        self.state = np.append(self.state, self.omega_k)  # 从节点是恶意节点时是否作恶
        self.state = np.append(self.state, self.T_1_1_mali)
        self.state = np.append(self.state, self.T_2_1_mali)  # 主节点是普通节点时作恶的时延
        self.state = np.append(self.state, self.T_4_k_mali)  # 从节点是普通节点时作恶的时延
        self.state = np.append(self.state, self.K_c_mali)  # 所有节点中恶意节点的个数
        self.state = np.append(self.state, np.ravel(self.a_i_k))
        return self._get_obs()

    def _get_obs(self):
        self.state = np.append(self.D_i_list, self.C_i_list)
        self.state = np.append(self.state, self.a_i_k)
        self.state = np.append(self.state, self.r_i_k)  # UE和SBS之间的链路速率
        self.state = np.append(self.state, self.U_k)  # 每个SBS的计算资源块的数量
        # self.state = np.append(self.state, self.r_m)  # UE和MBS之间的传输速率
        # self.state = np.append(self.state, self.f_m)  # MBS分配给每个任务的计算资源
        self.state = np.append(self.state, self.W_k_star)
        self.state = np.append(self.state, self.W_k_c)
        self.state = np.append(self.state, self.W_k)  # 时隙节点的信用权重
        self.state = np.append(self.state, self.s_k)  # 是否是恶意节点的标志
        self.state = np.append(self.state, self.N_fail)  # 时隙失败的次数
        self.state = np.append(self.state, self.T_block_ave)  # 平均共识时延
        self.state = np.append(self.state, self.omega_k_star)  # 主节点是恶意节点时是否作恶
        self.state = np.append(self.state, self.omega_k)  # 从节点是恶意节点时是否作恶
        self.state = np.append(self.state, self.T_1_1_mali)
        self.state = np.append(self.state, self.T_2_1_mali)  # 主节点是普通节点时作恶的时延
        self.state = np.append(self.state, self.T_4_k_mali)  # 从节点是普通节点时作恶的时延
        self.state = np.append(self.state, self.K_c_mali)  # 所有节点中恶意节点的个数
        return self.state

    # 工具函数,将[-1,1]映射为[0,1]
    def mapping(self, action):
        # action  = action[0]
        for i in range(len(action)):
            action[i] = (action[i] + 1.) * 0.5
        return action

    # 工具函数 ，状态 ue和SBS之间的链路可用性变成0，1变量
    def init(self, a_i_k):
        for i in range(self.I):
            for j in range(self.K):
                if a_i_k[i][j] < 0.5:
                    a_i_k[i][j] = 0
                else:
                    a_i_k[i][j] = 1

    # 是否是恶意节点的标志 （状态9）,也是公式9
    def my_method(self):
        global W_k  # 引用全局变量W_k
        # W_k = 0
        for i in range(self.K):
            if self.W_k <= 0.3:  # 公式9
                self.s_k[i] = 1
                # print("s_k3=", self.s_k)
            else:
                self.s_k[i] = 0
                # print("s_k4=", self.s_k)

    def mali_tolerance(self, K, s_k):
        K_c_mali = []
        for k in K:
            K_c_mali.append(1 - s_k[k])  # 公式12
        return K_c_mali

    def step(self, actions):
        # 1.用户任务卸载 2.像素的选择 3.任务处理中计算资源块的分配 4.传输速率分配 5.客户端的选择 6共识主节点的选择  7非主节点的选择 28dim
        # 1 用户任务卸载
        alpha = actions[:self.I]  # 用户的任务卸载（动作1）
        self.mapping(alpha)
        alpha1 = []
        for i in range(len(alpha)):
            if alpha[i] >= 0.5:
                alpha1.append(1.)
            else:
                alpha1.append(0.)
            # ap = np.reshape(alpha1,(self.I,3))

        # 2.任务处理中计算资源块的分配   （参考欢哥动作5写的）
        miu_i_k = actions[2 * self.I: 3 * self.I]
        self.mapping(miu_i_k)
        miu_i_k1 = []
        for i in range(self.I):
            if miu_i_k[i] >= 0.5:
                miu_i_k1.append(1.)
            else:
                miu_i_k1.append(0.)

        # 3.像素的选择
        s_i = actions[self.I: 2 * self.I]
        s_i = self.mapping(s_i)
        selected_resolutions = []  # 保存每个设备选择的分辨率
        num_options = len(self.resolution_options)
        for k in range(len(s_i)):
            # 将映射值乘以选项数量，并取 floor 得到离散索引
            idx = int(np.floor(s_i[k] * num_options))
            # 防止索引越界（当映射值恰好为1时）
            # 若 idx < 0 或 idx >= num_options，就修正
            if idx < 0:
                idx = 0
            if idx >= num_options:
                idx = num_options - 1

            # 记录选择的分辨率（(width, height)）
            selected_resolutions.append(self.resolution_options[idx])
        # print("Selected resolutions:", selected_resolutions)

        # 4传输速率的分配
        r_i_k = actions[3 * self.I: 3 * self.I + self.K]
        self.mapping(r_i_k)
        r_i_k1 = []
        for i in range(self.I):
            if r_i_k[i] >= 0.5:
                r_i_k1.append(1.)
            else:
                r_i_k1.append(0.)

        # 5. 客户端的选择
        z_k = actions[3 * self.I + self.K: 3 * self.I + 2 * self.K]
        z_k1 = self.mapping(z_k)
        z_k2 = []
        for k in range(self.K):
            # 如果 z_k1 的长度比 self.K 小，就会在这里报 index out of range
            if z_k1[k] >= 0.5:
                z_k2.append(1.)
            else:
                z_k2.append(0.)

        # print("z_k2 =", z_k2)

        # 6共识主节点的选择
        y_k_star = actions[3 * self.I + 2 * self.K: 3 * self.I + 3 * self.K]
        y_k_star1 = self.mapping(y_k_star)
        y_k_star2 = []
        for k in range(self.K):
            if y_k_star1[k] >= 0.5:
                y_k_star2.append(1.)
            else:
                y_k_star2.append(0.)

        # 7，共识非主节点的选择
        y_k = actions[3 * self.I + 3 * self.K: 3 * self.I + 4 * self.K]
        self.mapping(y_k)
        y_k1 = []  # M维的向量
        for i in range(len(y_k)):
            if y_k[i] >= 0.6:
                y_k1.append(1.)
            else:
                y_k1.append(0.)

        self.init(self.a_i_k)
        C = self.cost_t_e(alpha1, miu_i_k1, z_k2, y_k_star2, y_k1)
        Object = C[0]  # 计算cost

        # 约束4
        CRB_num_satisfy = True
        constrains_4 = 0
        for i in range(self.I):
            constrains_4 += alpha1[i] * min(miu_i_k1)
            if constrains_4 > self.U_k[i]:
                CRB_num_satisfy = False
        # 约束5
        # cli_selection_satisfy = False
        # if len(z_k2) > 0:
        #     for k in range(self.K):
        #         constrains_5 = 0
        #         constrains_5 += z_k2[0]
        #         if constrains_5 == 1:
        #             cli_selection_satisfy = True
        #             break
        cli_selection_satisfy = False
        if len(z_k2) > 0:
            for i in range(self.K):
                if z_k2[i] == 1:
                    cli_selection_satisfy = True
                    break

        # 约束6
        # pri_selection_satisfy = False
        # if len(y_k_star2) > 0:
        #     for k in range(self.K):
        #         constrains_6 = 0
        #         constrains_6 += y_k_star2[0]
        #         if constrains_6 == 1:
        #             pri_selection_satisfy = True
        #             break
        # 约束6
        pri_selection_satisfy = False
        if len(y_k_star2) > 0:
            for k in range(self.K):
                if y_k_star2[k] == 1:
                    pri_selection_satisfy = True
                    break

        con_node_satisfy = True
        for k in range(self.K):
            constrains_7 = 0
            constrains_7 += y_k_star2[0] * y_k1[0]
            if constrains_7 != 0:
                con_node_satisfy = False

        # 保证在同一时刻，一个节点不能同时是主节点和客户端
        cli_primary_satisfy = True
        for k in range(self.K):
            if z_k2[k] == 1 and y_k_star2[k] == 1:
                cli_primary_satisfy = False
                break

        # 约束8（公式11）
        mali_node_num = True
        for k in range(self.K):
            constrains_8 = 0
            constrains_8 = sum([self.s_k[k] for k in self.K]) > (self.K - 1) / 3
            if constrains_8 == True:
                mali_node_num = False

            # print("s_k6=", self.s_k)
            reward = 0

        # 约束11
        time_delay_satisfy = True
        constrains_11 = 0
        C = self.cost_t_e(alpha1, miu_i_k1, z_k2, y_k_star2, y_k1)
        T_task = C[5]
        T_block = C[1]
        constrains_11 = T_task + T_block
        # print(f"约束11的值为：{constrains_11}")
        if constrains_11 > self.delta_t:
            time_delay_satisfy = False

        # print(f"s_k={self.s_k}")
        if mali_node_num \
                and CRB_num_satisfy is False or cli_selection_satisfy is False or pri_selection_satisfy is False \
                or con_node_satisfy is False or cli_primary_satisfy is False and time_delay_satisfy is False:
            reward = 0  # 约束C4，C8，

            # 更新下一时刻状态
            self.C_i_list = self.renew_C_i_list()
            self.D_i_list = self.renew_D_i_list()
            self.a_i_k = self.renew_a_i_k()
            self.r_i_k = self.renew_r_i_k()
            self.U_k = self.renew_U_k()
            # self.r_m = self.renew_r_m()
            # self.f_m = self.renew_f_m()
            self.W_k_star = self.renew_W_k_star()
            self.W_k_c = self.renew_W_k_c()
            self.W_k = self.renew_W_k()
            self.s_k = self.renew_s_k()
            self.K_c_mali = self.renew_K_c_mali()
            self.omega_k_star = self.calculate_omega_k_stars()
            self.omega_k = self.calculate_omega_k()
            self.T_1_1_mali = self.renew_T_1_1_mali()
            self.T_2_1_mali = self.renew_T_2_1_mali()
            self.T_4_k_mali = self.renew_T_4_k_mali()
            self.N_fail = self.renew_N_fail(alpha1, miu_i_k1, z_k2, y_k_star2, y_k1)
            self.T_block_ave = self.renew_T_block_ave(alpha1, miu_i_k1, z_k2, y_k_star2, y_k1)
        else:
            Object = np.min(Object)
            reward = 1 / (Object + 1e-5)

            # 跟新下一时刻状态
            self.C_i_list = self.renew_C_i_list()
            self.D_i_list = self.renew_D_i_list()
            self.a_i_k = self.renew_a_i_k()
            self.r_i_k = self.renew_r_i_k()
            self.U_k = self.renew_U_k()
            # self.r_m = self.renew_r_m()
            # self.f_m = self.renew_f_m()
            self.W_k_star = self.renew_W_k_star()
            self.W_k_c = self.renew_W_k_c()
            self.W_k = self.renew_W_k()
            self.s_k = self.renew_s_k()
            # print("s_k8=", self.s_k)
            self.K_c_mali = self.renew_K_c_mali()
            self.omega_k_star = self.calculate_omega_k_stars()
            self.omega_k = self.calculate_omega_k()
            self.T_1_1_mali = self.renew_T_1_1_mali()
            self.T_2_1_mali = self.renew_T_2_1_mali()
            self.T_4_k_mali = self.renew_T_4_k_mali()
            self.N_fail = self.renew_N_fail(alpha1, miu_i_k1, z_k2, y_k_star2, y_k1)
            self.T_block_ave = self.renew_T_block_ave(alpha1, miu_i_k1, z_k2, y_k_star2, y_k1)

        # s_ = self._get_obs()
        return reward, self._get_obs()

    # 更新状态1  任务工作量
    def renew_C_i_list(self):
        C_i_list = self.task_density * self.D_i_list
        return C_i_list

    # 更新状态2 输入任务的数据大小
    def renew_D_i_list(self):
        # (width, height) = self.resolution_options
        D_i_list = np.random.randint(5, 8, self.I) * 10 ** 5
        return D_i_list

    # 更新状态3  用户和SBS之间的是否链接
    def renew_a_i_k(self):
        a_i_k = np.array(np.random.randint(0, 2, size=[self.I, self.K]), dtype='int64')
        return a_i_k

    # 更新状态4  UE和SBS之间链路速率
    def renew_r_i_k(self):
        r_i_k = np.random.randint(5, 10, self.I) * 10 ** 6
        return r_i_k

    # 更新状态5 每个SBS的计算资源块的数量
    def renew_U_k(self):
        U_k = np.random.randint(20, 51, self.K)
        return U_k

    # 更新状态6 UE和MBS之间的传输速率
    # def renew_r_m(self):
    #     r_m = np.random.randint(30, 50, self.I) * 10 ** 6
    #     return r_m

    # 更新状态7 MBS分配给每个任务的计算资源
    # def renew_f_m(self):
    #     f_m = np.random.randint(50, 101, self.M) * 10 ** 6
    #     return f_m

    def renew_W_k_c(self):
        if self.W_k_c > self.Thre_h_w:
            W_k_c = min(self.W_k_c * self.delta_cli, 0.3)
        else:
            W_k_c = (self.W_k_c * self.delta_cli * self.lambda1) / (self.T_1_1_mali + 1e-5)
        return W_k_c

    def renew_W_k_star(self):
        if self.W_k_star > self.Thre_h_w:
            W_k_star = min(self.W_k_star * self.delta_pri, 0.3)
        else:
            W_k_star = (self.W_k_star * self.delta_pri * self.lambda1) / (self.T_2_1_mali + 1e-5)
        return W_k_star

    # 更新状态8 主节点的信用权重（公式22）
    def renew_W_k(self):
        if self.T_2_1_mali > self.Thre_h_w:
            W_k = min(self.W_k * self.delta_con, 0.3)
        else:
            W_k = self.W_k * self.delta_con * (
                    1 / self.T_2_1_mali + 1e-5) * self.lambda1  # 最后一个1为统一单位的参数，也是为了使延迟和权重保持在同一量级
        return W_k

    # 更新状态9 是否恶意节点的标志（根据 Wk(t) 的值，根据公式9更新）
    def renew_s_k(self):
        global W_k  # 引用全局变量W_k

        for i in range(self.K):
            if self.W_k <= 0.3:  # 公式9
                self.s_k[i] = 1
                # print("s_k9=", self.s_k)
            else:
                self.s_k[i] = 0
                # print("s_k10=", self.s_k)
        return self.s_k

    # 更新状态10 所有节点中恶意节点的数量
    def renew_K_c_mali(self):
        K_c_mali = []
        for k in range(self.K):
            K_c_mali.append(self.s_k[k])
        return K_c_mali

    # 更新状态11 主节点是恶意节点时是否作恶
    def calculate_omega_k_stars(self):
        def renew_omega_k_star(w):
            if random.random() < w:
                return 1
            else:
                return 0

        self.W_k_star = 0.3
        omega_k_stars = [renew_omega_k_star(1 - self.W_k_star) for _ in range(1)]
        return omega_k_stars

    # 更新状态12 从节点是恶意节点时是否作恶
    def calculate_omega_k(self):
        def renew_omega_k(w):
            if random.random() < w:
                return 1
            else:
                return 0

        renew_omega_k = [renew_omega_k(1 - self.W_k) for _ in range(1)]
        return renew_omega_k

    # 更新状态13 主节点是普通节点时作恶的延迟(根据公式40)
    # def renew_T_1_1_mali(self):
    #     self.T_1_1_mali = np.max((1 - self.s_k) * (1 / self.W_k_c + 1e-5) * (1 / 20) * self.delta_t)
    #     return self.T_1_1_mali
    def renew_T_1_1_mali(self):
        if np.any(self.W_k_c == 0):
            return 0
        self.T_1_1_mali = np.max((1 - self.s_k) * (1 / (self.W_k_c + 1e-5)) * (1 / 20) * self.delta_t)
        return self.T_1_1_mali

    def renew_T_2_1_mali(self):
        self.T_2_1_mali = np.max((1 - self.s_k) * (1 / self.W_k_star + 1e-5) * (1 / 20) * self.delta_t)
        return self.T_2_1_mali

    # 更新状态14 从节点是普通节点时作恶的延迟（根据公式41）
    def renew_T_4_k_mali(self):
        self.T_4_k_mali = np.max((1 - self.s_k) * (1 / self.W_k + 1e-5) * (1 / 20) * self.delta_t)
        return self.T_4_k_mali

    # 更新状态15 时隙失败的次数(根据公式30)
    def renew_N_fail(self, alpha1, miu_i_k1, z_k2, y_k_star2, y_k1):
        C = self.cost_t_e(alpha1, miu_i_k1, z_k2, y_k_star2, y_k1)
        for k in range(self.I):
            T_task = C[5]
            T_block = C[1]
            judge_con = int(C[3] == self.delta_t) or int(C[4] >= self.delta_t) or int(T_block >= self.delta_t) or int(
                T_task >= self.delta_t) or int(T_task + T_block > self.delta_t) \
                        or int(1 - self.s_k[k] > (self.K - 1) / 3)  # 公式30的判断条件
            N_fail = self.N_fail + judge_con
            return N_fail

    # 更新状态16 平均共识时延
    def renew_T_block_ave(self, alpha1, miu_i_k1, z_k2, y_k_star2, y_k1):
        C = self.cost_t_e(alpha1, miu_i_k1, z_k2, y_k_star2, y_k1)
        T_block = C[1]
        self.T_block_ave = (1 / self.delta_t) * T_block
        return self.T_block_ave

    # 计算cost
    def cost_t_e(self, alpha1, miu_i_k1, z_k2, y_k_star2, y_k1):
        T_i = []
        T_i_k = []
        for i in range(self.I):
            result1 = []
            for k in range(self.K):
                tmp = [self.C_i_list[i] / (miu_i_k1[j] * self.f_s) for j in range(self.I)]

                result2 = self.D_i_list[i] / (self.r_i_k[i] + 1e-5) + sum(tmp)
                result2 = result2.tolist()
                result1.append(result2)

            t_i_s = np.dot(self.a_i_k[i], result1)  # 点乘
            T_i.append(alpha1[i] * t_i_s)

        T_task = max(T_i)  # 公式8

        self.my_method()

        K_c_mali = []
        for k in range(self.K):
            K_c_mali.append(self.s_k[k])  # 公式12
            # K_c_mali_1.append(K_c_mali)
        K_c_mali = sum(K_c_mali)

        # D区块链共识延迟模型
        # Request阶段的计算周期（公式13）
        # T_1 = (self.theta + self.sigma) * self.I / (self.U_k_star * self.f_s)
        # T_1 = np.max(T_1)
        T_1_1_C = (self.theta + self.sigma) * self.I / (self.U_k * self.f_s) + self.omega_k * self.T_1_1_mali
        T_1_1_C = np.max(T_1_1_C)

        T_1_2_star = (self.theta + self.sigma) * self.I / (self.U_k * self.f_s)
        T_1 = T_1_1_C + T_1_2_star
        # pre - prepare阶段延迟
        # 这里要用一个if语句来写
        # 如果主节点是正常工作，T_21为下式

        # 如果主节点是一般节点，但它作恶并且延迟是T_21_mali∈[0，τ]
        T_2_1 = []
        T_2_2 = []
        for k in range(self.K):
            T_2_11 = (self.theta + (self.K - 1) * self.sigma) / (self.U_k_star[k] * self.f_s) + y_k_star2[k] * (
                    1 - self.s_k[k]) * self.T_2_1_mali  # 公式14
            T_2_1.append(T_2_11)

        for k in range(self.I):
            T_non_pri = (self.I + 1) * (self.theta + self.sigma) / (
                    self.U_k[k] * self.f_s + 1e-5)  # 非主节点的验证延迟为T_non_pri
            T_2_22 = T_non_pri  # 公式15
            T_2_2.append(T_2_22)
        T_2 = T_2_1 + T_2_2  # （公式16）
        T_2 = min(T_2)

        # prepare阶段延迟
        T_3_non = (self.theta + (self.K - 1) * self.sigma + (2 * K_c_mali + 1) * (self.theta + self.sigma)) / (
                self.U_k * self.f_s + 1e-5)  # 公式17（非主节点K的延迟）
        T_3_non = max(T_3_non)
        T_3_star = ((2 * K_c_mali + 1) * (self.theta + self.sigma)) / (
                self.U_k_star * self.f_s)  # 公式18（主节点k的延迟）
        T_3_star = max(T_3_star)
        T_3 = max(T_3_star, T_3_non)  # 公式19

        # commit阶段延迟
        T_4_non1 = []
        for k in range(self.K):
            if y_k1[k] == 1:  # 非主节点是恶意节点,并执行
                T_4_non = self.delta_t
            elif y_k1[k] >= 0 and y_k1[k] < 1:  # 非主节点是一般节点
                if y_k1[k] < 0.5:  # 非主节点是一般节点但作恶
                    T_4_non = (self.theta + self.sigma) / (self.U_k * self.f_s + 1e-5) + self.T_4_k_mali
                else:  # 非主节点是一般节点，不作恶
                    T_4_non = (self.theta + self.sigma) / (self.U_k * self.f_s + 1e-5)
            T_4_non1.append(T_4_non)
        # print("4",T_4_non1)

        found = False
        i = 0
        K = []
        T_4 = 0
        while not found and i < len(K):
            if K[i] == 2 * K_c_mali:
                T_4 = T_4_non1
                found = True  # 公式20
            # i += 1
            else:
                T_4 = 1 / 2 * self.theta  # 这里是自己随便编的值
            i += 1
            '''其中，found是一个标志变量，用于标记是否已经找到了符合条件的非主节点。
            一开始将其初始化为False。i表示当前遍历到的非主节点的索引，开始时将其初始化为0。
            在while循环中，首先判断是否已经找到了符合条件的非主节点，如果没有，则继续循环。
            在每次循环中，判断当前遍历到的非主节点的编号是否等于2K_c_mali，如果是，则将T4的值设置为T_4_non，
            同时将found标记为True，表示已经找到符合条件的非主节点。
            最后，每次循环结束后，将i加1，继续下一次循环。'''
        T_2 = abs(T_2)
        # T_block = T_1 + T_2 + T_3 + T_4  # 公式21
        T_block = max(0, T_1 + T_2 + T_3 + T_4)

        # 公式26
        T_block_ave = (1 / self.delta_t) * T_block

        # 公式30
        # for k in range(self.I):
        #     judge_con = int(T_2 == self.delta_t) or int(T_4 >= self.delta_t) or int(T_block >= self.delta_t) or int(
        #         T_task >= self.delta_t) or int(T_task + T_block > self.delta_t) \
        #                 or int(1 - self.s_k[k] > (self.K - 1) / 3)  # 公式30的判断条件
        #     N_fail = self.N_fail + judge_con

        for k in range(self.I):
            judge_con = int(T_task+ T_block >= self.delta_t)
            N_fail = self.N_fail + judge_con
            R_fail = N_fail / self.delta_t  # 公式31
        # int(condition) 表示当条件condition满足时返回1，否则返回0。

        # 目标函数
        Object = []
        for i in range(self.I):
            Object1 = R_fail  # 公式32
            Object.append(Object1)
        Object = min(Object) * 1 / (self.T)

        return Object, T_block, T_i, T_2, T_4, T_task
