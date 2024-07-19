import numpy as np
import random
from vprtw_aco_figure import VrptwAcoFigure
from vrptw_base import VrptwGraph, PathMessage
from ant import Ant
from threading import Thread
from queue import Queue
import time


class BasicACO:
    def __init__(self, graph: VrptwGraph, ants_num=10, max_iter=200, beta=2, q0=0.1,
                 whether_or_not_to_show_figure=True):
        super()
        # graph posisi node dan informasi waktu layanan
        self.graph = graph
        # ants_num jumlah semut
        self.ants_num = ants_num
        # max_iter jumlah iterasi maksimal
        self.max_iter = max_iter
        # beta pentingnya informasi heuristik
        self.beta = beta
        # q0 probabilitas memilih titik berikutnya dengan probabilitas tertinggi
        self.q0 = q0
        # best path
        self.best_path_distance = None
        self.best_path = None
        self.best_vehicle_num = None

        self.whether_or_not_to_show_figure = whether_or_not_to_show_figure

    def run_basic_aco(self):
        """
        Memulai thread lain untuk menjalankan basic_aco, menggunakan thread utama untuk menggambar
        :return:
        """
        path_queue_for_figure = Queue()
        basic_aco_thread = Thread(target=self._basic_aco, args=(path_queue_for_figure,))
        basic_aco_thread.start()

        # Apakah akan menampilkan gambar
        if self.whether_or_not_to_show_figure:
            figure = VrptwAcoFigure(self.graph.nodes, path_queue_for_figure)
            figure.run()
        basic_aco_thread.join()

        # Mengirimkan None sebagai tanda akhir
        if self.whether_or_not_to_show_figure:
            path_queue_for_figure.put(PathMessage(None, None))

    def _basic_aco(self, path_queue_for_figure: Queue):
        """
        Algoritma koloni semut dasar
        :return:
        """
        start_time_total = time.time()

        # Jumlah iterasi maksimal
        start_iteration = 0
        for iter in range(self.max_iter):

            # Mengatur muatan kendaraan, jarak tempuh, dan waktu perjalanan untuk setiap semut
            ants = list(Ant(self.graph) for _ in range(self.ants_num))
            for k in range(self.ants_num):

                # Semut harus mengunjungi semua pelanggan
                while not ants[k].index_to_visit_empty():
                    next_index = self.select_next_index(ants[k])
                    # Memeriksa apakah bergabung dengan posisi tersebut masih memenuhi batasan, jika tidak maka memilih lagi, lalu memeriksa lagi
                    if not ants[k].check_condition(next_index):
                        next_index = self.select_next_index(ants[k])
                        if not ants[k].check_condition(next_index):
                            next_index = 0

                    # Memperbarui jalur semut
                    ants[k].move_to_next_index(next_index)
                    self.graph.local_update_pheromone(ants[k].current_index, next_index)

                # Kembali ke 0 di akhir
                ants[k].move_to_next_index(0)
                self.graph.local_update_pheromone(ants[k].current_index, 0)

            # Menghitung jarak tempuh untuk semua semut
            paths_distance = np.array([ant.total_travel_distance for ant in ants])

            # Merekam jalur terbaik saat ini
            best_index = np.argmin(paths_distance)
            if self.best_path is None or paths_distance[best_index] < self.best_path_distance:
                self.best_path = ants[int(best_index)].travel_path
                self.best_path_distance = paths_distance[best_index]
                self.best_vehicle_num = self.best_path.count(0) - 1
                start_iteration = iter

                # Menampilkan gambar
                if self.whether_or_not_to_show_figure:
                    path_queue_for_figure.put(PathMessage(self.best_path, self.best_path_distance))

                print('\n')
                print('[iterasi %d]: menemukan jalur yang lebih baik, jaraknya adalah %f' % (iter, self.best_path_distance))
                print('waktu yang dibutuhkan %0.3f detik menjalankan multiple_ant_colony_system' % (time.time() - start_time_total))

            # Memperbarui matriks feromon
            self.graph.global_update_pheromone(self.best_path, self.best_path_distance)

            given_iteration = 100
            if iter - start_iteration > given_iteration:
                print('\n')
                print('keluar iterasi: tidak dapat menemukan solusi yang lebih baik dalam %d iterasi' % given_iteration)
                break

        print('\n')
        print('jarak jalur terbaik akhir adalah %f, jumlah kendaraan adalah %d' % (self.best_path_distance, self.best_vehicle_num))
        print('waktu yang dibutuhkan %0.3f detik menjalankan multiple_ant_colony_system' % (time.time() - start_time_total))

    def select_next_index(self, ant):
        """
        Memilih node berikutnya
        :param ant:
        :return:
        """
        current_index = ant.current_index
        index_to_visit = ant.index_to_visit

        transition_prob = self.graph.pheromone_mat[current_index][index_to_visit] * \
            np.power(self.graph.heuristic_info_mat[current_index][index_to_visit], self.beta)
        transition_prob = transition_prob / np.sum(transition_prob)

        if np.random.rand() < self.q0:
            max_prob_index = np.argmax(transition_prob)
            next_index = index_to_visit[max_prob_index]
        else:
            # Menggunakan algoritma roulette wheel
            next_index = BasicACO.stochastic_accept(index_to_visit, transition_prob)
        return next_index

    @staticmethod
    def stochastic_accept(index_to_visit, transition_prob):
        """
        Roulette wheel
        :param index_to_visit: daftar N indeks (daftar atau tuple)
        :param transition_prob:
        :return: indeks yang dipilih
        """
        # menghitung N dan nilai kebugaran maksimum
        N = len(index_to_visit)

        # normalisasi
        sum_tran_prob = np.sum(transition_prob)
        norm_transition_prob = transition_prob/sum_tran_prob

        # memilih: O(1)
        while True:
            # memilih individu secara acak dengan probabilitas uniform
            ind = int(N * random.random())
            if random.random() <= norm_transition_prob[ind]:
                return index_to_visit[ind]