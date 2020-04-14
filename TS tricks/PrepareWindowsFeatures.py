import pandas as pd
import numpy as np
from tqdm.auto import tqdm


class WindowFeatures:
    """
    класс содержащий в себе всё что требуется для получения оконных признаков
    """
    def __init__(self, windows, funcs, funcs_names, relative_flag = True, runs_flag = True):
        """
        windows - список окон на которых требуется построить признаки
        funcs - список функциия применяемых на окне
        funcs_names - наименования оконных функций
        relative_flag - флаг необходимости построения относительных признаков
        runs_flag - флаг 
        """
        
        self.windows = sorted(windows).copy()
        self.funcs = funcs.copy()
        self.funcs_names = funcs_names.copy()
        self.relative_flag = relative_flag
        self.runs_flag = runs_flag
        
        
    def get_window_features(self, df, cols):
        """
        функция возвращающая исходный датафрейм с признаками
        полученными с помощью функций self.funcs на окнах self.windows
        
        df - исходный датафрейм с данными
        col - рассматриваемый параметр
        """
        
        
        #1. Заполним пропуски
        for col in cols:
            df[col].ffill(inplace = True)
            
            
        #2. 
        if not self.runs_flag:
            df['Номер пробега'] = 0
            df['Номер пробега'] = df['Номер пробега'].astype(int)
        
        
        #3. Получим оконные признаки
        return self.__calculate_windows_stats__(df, cols)
        
        
        
    def __window_stat_run__(self, arr, stat, window_size):
        '''
        функция расчета статистики stat для окна window_size
        arr - numpy матрица с сырыми признаками
        stat - расчитываемая статистики
        window_size - окно на котором считается статистика
        '''

        arr_stat = np.zeros_like(arr) * np.nan
        for i in range(arr.shape[0]):
            up_ind = i - window_size + 1
            if up_ind >= 0:
                down_ind = i+1
                arr_stat[i, :] = stat(arr[up_ind : down_ind, :])
            else:
                arr_stat[i, :] = np.nan

        return arr_stat


    def __windows_stats_run__(self, arr):
        """
        функция расчета статистик на окнах для одного пробега
        arr - numpy матрица с сырыми признаками одного для одного признака
        """
        
        funcs_dim = len(self.funcs)
        arr_stats = np.zeros((arr.shape[0], arr.shape[1]*funcs_dim*len(self.windows)))*np.nan
        
        for i, window_size in enumerate(self.windows):
            for j, stat in enumerate(self.funcs):
                left_ind = i*arr.shape[1]*funcs_dim + j*arr.shape[1]
                right_ind = i*arr.shape[1]*funcs_dim + (j+1)*arr.shape[1]
                arr_stats[:, left_ind : right_ind] = self.__window_stat_run__(arr, stat, window_size)
        return arr_stats   
        
        
    
    def __calculate_relative_features__(self, df, cols, arr_stats):
        """
        функция расчета относительных признаков
        cols - колонки для которых проводятся рассчеты
        arr_stats - numpy матрица с оконными признаками
        """
    
        # 1. Выделение памяти под относительные признаки
        windows_dim = len(self.windows)
        faxis = windows_dim*(windows_dim-1)*len(cols)*len(self.funcs)//2
        arr_stats_relative = np.zeros((df.shape[0], faxis)) * np.nan
        columns_name_arr_stats_relative = []
        t = 0


        # 2. Перебираем окна (формат перебора парой левое-правое окно)
        for i in range(windows_dim - 1):
            for j in range(i+1, windows_dim):
                # Границы левого окна
                window_i_l = arr_stats.shape[1]//windows_dim * i
                window_i_r = arr_stats.shape[1]//windows_dim * (i+1)

                # Границы правого окна
                window_j_l = arr_stats.shape[1]//windows_dim * j
                window_j_r = arr_stats.shape[1]//windows_dim * (j+1)

                # Границы arr_stats_relative, в него записываются относительные статистики
                t_l = arr_stats.shape[1]//windows_dim * t
                t_r = arr_stats.shape[1]//windows_dim * (t+1)


                arr_stats_relative[:, t_l:t_r] = arr_stats[:, window_i_l:window_i_r] /\
                                                (arr_stats[:, window_j_l:window_j_r] + np.finfo(np.float32).eps)
                t += 1

                # Кастанем к np.nan начала относительных окон 
                arr_stats_relative[:min(self.windows[i], arr_stats_relative.shape[0]), t_l:t_r] = np.nan


                # Дадим имена колонкам
                for name_stat in self.funcs_names:
                    for column in cols:
                        name_col = '{}_{}_{}/{}'.format(column, name_stat, self.windows[i], self.windows[j])
                        columns_name_arr_stats_relative += [name_col]

        return arr_stats_relative, columns_name_arr_stats_relative
        
        
        
    def __calculate_windows_stats__(self, df, cols):
        """
        функция расчета оконных признаков и конкатинации их к исходным данным
        df - исходные данные
        cols - колонки для которых требуется рассчитать признаки
        """


        # 1. Выделение памяти под полученные признаки
        arr_stats = np.zeros((df.shape[0], len(cols)*len(self.funcs)*len(self.windows)))*np.nan


        # 2. Преобразование исходных данных в numpy array
        arr = np.array(df[cols])
        nruns = df['Номер пробега'].values
        uniq_nruns = np.unique(nruns)



        # 3. Рассчет статистик для каждого пробега
        for nrun in tqdm(uniq_nruns):
            mask_run = nruns == nrun
            arr_stats[mask_run, :] = self.__windows_stats_run__(arr[mask_run, :])


        # 4. Получение название колонок
        columns_name_arr_stats = []
        for window_size in self.windows:
            for name_stat in self.funcs_names:
                for column in cols:
                    name_col = '{}_{}_{}'.format(column, name_stat, window_size)
                    columns_name_arr_stats += [name_col]


        # 5. Расчет относительных признаков 
        if self.relative_flag:
            arr_stats_relative, columns_name_arr_stats_relative = self.__calculate_relative_features__(df, cols, arr_stats)
                                                                                                  
            arr_stats_relative = pd.DataFrame(data=arr_stats_relative, columns=columns_name_arr_stats_relative)


        # 6. Конкатинация данных
        df.reset_index(inplace=True, drop=True)
        arr_stats = pd.DataFrame(data=arr_stats, columns=columns_name_arr_stats)

        if self.relative_flag:
            df = pd.concat([df, arr_stats, arr_stats_relative], axis=1)
        else:
            df = pd.concat([df, arr_stats], axis=1)


        return df