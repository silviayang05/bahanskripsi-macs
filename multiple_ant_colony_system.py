import numpy as np
import random
from vprtw_aco_figure import VrptwAcoFigure
from vrptw_base import VrptwGraph, PathMessage
from ant import Ant
from threading import Thread, Event
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import copy
import time
from multiprocessing import Process
from multiprocessing import Queue as MPQueue


class MultipleAntColonySystem:
    def __init__(self, graph: VrptwGraph, ants_num=10, beta=1, q0=0.1, whether_or_not_to_show_figure=True):
        super()
        # graph posisi node dan informasi waktu layanan
        self.graph = graph
        # ants_num jumlah semut
        self.ants_num = ants_num
        # vehicle_capacity kapasitas maksimal kendaraan
        self.max_load = graph.vehicle_capacity
        # beta pentingnya informasi heuristik
        self.beta = beta
        # q0 probabilitas memilih titik berikutnya dengan probabilitas tertinggi
        self.q0 = q0
        # best path
        self.best_path_distance = None
        self.best_path = None
        self.best_vehicle_num = None

        self.whether_or_not_to_show_figure = whether_or_not_to_show_figure

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

    @staticmethod
    def new_active_ant(ant: Ant, vehicle_num: int, local_search: bool, IN: np.numarray, q0: float, beta: int, stop_event: Event, max_iterations: int):
        """
        Menjelajahi peta dengan jumlah kendaraan yang ditentukan, jumlah kendaraan yang digunakan tidak boleh lebih dari jumlah yang ditentukan
        :param ant:
        :param vehicle_num:
        :param local_search:
        :param IN:
        :param q0:
        :param beta:
        :param stop_event:
        :param max_iterations: Jumlah iterasi maksimal
        :return:
        """
        # print('[new_active_ant]: mulai, indeks awal %d' % ant.travel_path[0])

        # Dalam new_active_ant, maksimal dapat menggunakan vehicle_num kendaraan, yaitu maksimal dapat mencakup vehicle_num+1 node depot, karena node awal sudah digunakan, jadi hanya tersisa vehicle depot
        unused_depot_count = vehicle_num

        # Jika masih ada node yang belum dikunjungi dan masih bisa kembali ke depot
        while not ant.index_to_visit_empty() and unused_depot_count > 0:
            if stop_event.is_set():
                # print('[new_active_ant]: menerima event stop')
                return

            # Menghitung semua node berikutnya yang memenuhi batasan beban dan lainnya
            next_index_meet_constrains = ant.cal_next_index_meet_constrains()

            # Jika tidak ada node berikutnya yang memenuhi batasan, kembali ke depot
            if len(next_index_meet_constrains) == 0:
                ant.move_to_next_index(0)
                unused_depot_count -= 1
                continue

            # Mulai menghitung probabilitas memilih setiap node berikutnya yang memenuhi batasan
            length = len(next_index_meet_constrains)
            ready_time = np.zeros(length)
            due_time = np.zeros(length)

            for i in range(length):
                ready_time[i] = ant.graph.nodes[next_index_meet_constrains[i]].ready_time
                due_time[i] = ant.graph.nodes[next_index_meet_constrains[i]].due_time

            delivery_time = np.maximum(ant.vehicle_travel_time + ant.graph.node_dist_mat[ant.current_index][next_index_meet_constrains], ready_time)
            delta_time = delivery_time - ant.vehicle_travel_time
            distance = delta_time * (due_time - ant.vehicle_travel_time)

            distance = np.maximum(1.0, distance-IN[next_index_meet_constrains])
            closeness = 1/distance

            transition_prob = ant.graph.pheromone_mat[ant.current_index][next_index_meet_constrains] * \
                              np.power(closeness, beta)
            transition_prob = transition_prob / np.sum(transition_prob)

            # Memilih secara langsung titik dengan closeness terbesar
            if np.random.rand() < q0:
                max_prob_index = np.argmax(transition_prob)
                next_index = next_index_meet_constrains[max_prob_index]
            else:
                # Menggunakan algoritma roulette wheel
                next_index = MultipleAntColonySystem.stochastic_accept(next_index_meet_constrains, transition_prob)

            # Memperbarui matriks feromon
            ant.graph.local_update_pheromone(ant.current_index, next_index)
            ant.move_to_next_index(next_index)

        # Jika sudah mengunjungi semua node, perlu kembali ke depot
        if ant.index_to_visit_empty():
            ant.graph.local_update_pheromone(ant.current_index, 0)
            ant.move_to_next_index(0)

        # Memasukkan node yang belum dikunjungi untuk memastikan path adalah feasible
        ant.insertion_procedure(stop_event, max_iterations)

        # ant.index_to_visit_empty()==True berarti feasible
        if local_search is True and ant.index_to_visit_empty():
            ant.local_search_procedure(stop_event, max_iterations)

    @staticmethod
    def acs_time(new_graph: VrptwGraph, vehicle_num: int, ants_num: int, q0: float, beta: int,
                 global_path_queue: Queue, path_found_queue: Queue, stop_event: Event, max_iterations: int):
        """
        Untuk acs_time, perlu mengunjungi semua node (path adalah feasible), dan mencari jarak tempuh yang lebih pendek
        :param new_graph:
        :param vehicle_num:
        :param ants_num:
        :param q0:
        :param beta:
        :param global_path_queue:
        :param path_found_queue:
        :param stop_event:
        :param max_iterations: Jumlah iterasi maksimal
        :return:
        """

        # Maksimal dapat menggunakan vehicle_num kendaraan, yaitu mencari jarak tempuh terpendek dengan menggunakan vehicle_num+1 node depot
        # vehicle_num disetel sama dengan best_path saat ini
        print('[acs_time]: mulai, jumlah kendaraan %d' % vehicle_num)
        # Inisialisasi matriks feromon
        global_best_path = None
        global_best_distance = None
        ants_pool = ThreadPoolExecutor(ants_num)
        ants_thread = []
        ants = []
        while True:
            print('[acs_time]: iterasi baru')

            if stop_event.is_set():
                print('[acs_time]: menerima event stop')
                return

            for k in range(ants_num):
                ant = Ant(new_graph, 0)
                thread = ants_pool.submit(MultipleAntColonySystem.new_active_ant, ant, vehicle_num, True,
                                          np.zeros(new_graph.node_num), q0, beta, stop_event, max_iterations)
                ants_thread.append(thread)
                ants.append(ant)

            # Di sini dapat menggunakan metode result, menunggu thread selesai
            for thread in ants_thread:
                thread.result()

            ant_best_travel_distance = None
            ant_best_path = None
            # Memeriksa apakah semut menemukan path yang feasible dan lebih baik dari global path
            for ant in ants:

                if stop_event.is_set():
                    print('[acs_time]: menerima event stop')
                    return

                # Mendapatkan best path saat ini
                if not global_path_queue.empty():
                    info = global_path_queue.get()
                    while not global_path_queue.empty():
                        info = global_path_queue.get()
                    print('[acs_time]: menerima info jalur global')
                    global_best_path, global_best_distance, global_used_vehicle_num = info.get_path_info()

                # Jarak tempuh terpendek yang ditemukan oleh semut
                if ant.index_to_visit_empty() and (ant_best_travel_distance is None or ant.total_travel_distance < ant_best_travel_distance):
                    ant_best_travel_distance = ant.total_travel_distance
                    ant_best_path = ant.travel_path

            # Melakukan pembaruan feromon global di sini
            new_graph.global_update_pheromone(global_best_path, global_best_distance)

            # Mengirimkan path yang ditemukan oleh semut ke macs
            if ant_best_travel_distance is not None and ant_best_travel_distance < global_best_distance:
                print('[acs_time]: semut melakukan local search dan menemukan jalur feasible yang lebih baik, mengirim info jalur ke macs')
                path_found_queue.put(PathMessage(ant_best_path, ant_best_travel_distance))

            ants_thread.clear()
            for ant in ants:
                ant.clear()
                del ant
            ants.clear()

    @staticmethod
    def acs_vehicle(new_graph: VrptwGraph, vehicle_num: int, ants_num: int, q0: float, beta: int,
                    global_path_queue: Queue, path_found_queue: Queue, stop_event: Event, max_iterations: int):
        """
        Untuk acs_vehicle, jumlah kendaraan yang digunakan akan kurang satu dari jumlah kendaraan yang digunakan pada best path saat ini, gunakan lebih sedikit kendaraan, coba untuk mengunjungi lebih banyak node, jika sudah mengunjungi semua node (path adalah feasible), kirim ke macs
        :param new_graph:
        :param vehicle_num:
        :param ants_num:
        :param q0:
        :param beta:
        :param global_path_queue:
        :param path_found_queue:
        :param stop_event:
        :param max_iterations: Jumlah iterasi maksimal
        :return:
        """
        # vehicle_num disetel kurang satu dari best_path saat ini
        print('[acs_vehicle]: mulai, jumlah kendaraan %d' % vehicle_num)
        global_best_path = None
        global_best_distance = None

        # Menggunakan algoritma nearest_neighbor_heuristic untuk menginisialisasi path dan distance
        current_path, current_path_distance, _ = new_graph.nearest_neighbor_heuristic(max_vehicle_num=vehicle_num)

        # Menemukan node yang belum dikunjungi dalam current_path
        current_index_to_visit = list(range(new_graph.node_num))
        for ind in set(current_path):
            current_index_to_visit.remove(ind)

        ants_pool = ThreadPoolExecutor(ants_num)
        ants_thread = []
        ants = []
        IN = np.zeros(new_graph.node_num)
        while True:
            print('[acs_vehicle]: iterasi baru')

            if stop_event.is_set():
                print('[acs_vehicle]: menerima event stop')
                return

            for k in range(ants_num):
                ant = Ant(new_graph, 0)
                thread = ants_pool.submit(MultipleAntColonySystem.new_active_ant, ant, vehicle_num, False, IN, q0,
                                          beta, stop_event, max_iterations)

                ants_thread.append(thread)
                ants.append(ant)

            # Di sini dapat menggunakan metode result, menunggu thread selesai
            for thread in ants_thread:
                thread.result()

            for ant in ants:

                if stop_event.is_set():
                    print('[acs_vehicle]: menerima event stop')
                    return

                IN[ant.index_to_visit] = IN[ant.index_to_visit]+1

                # Jika semut menemukan path yang dapat mengunjungi lebih banyak node dengan menggunakan vehicle_num kendaraan
                if len(ant.index_to_visit) < len(current_index_to_visit):
                    current_path = copy.deepcopy(ant.travel_path)
                    current_index_to_visit = copy.deepcopy(ant.index_to_visit)
                    current_path_distance = ant.total_travel_distance
                    # Dan mengatur IN menjadi 0
                    IN = np.zeros(new_graph.node_num)

                    # Jika path ini adalah feasible, kirim ke macs_vrptw
                    if ant.index_to_visit_empty():
                        print('[acs_vehicle]: menemukan jalur feasible, mengirim info jalur ke macs')
                        path_found_queue.put(PathMessage(ant.travel_path, ant.total_travel_distance))

            # Memperbarui info di new_graph, global
            new_graph.global_update_pheromone(current_path, current_path_distance)

            if not global_path_queue.empty():
                info = global_path_queue.get()
                while not global_path_queue.empty():
                    info = global_path_queue.get()
                print('[acs_vehicle]: menerima info jalur global')
                global_best_path, global_best_distance, global_used_vehicle_num = info.get_path_info()

            new_graph.global_update_pheromone(global_best_path, global_best_distance)

            ants_thread.clear()
            for ant in ants:
                ant.clear()
                del ant
            ants.clear()

    def run_multiple_ant_colony_system(self, max_iterations: int, file_to_write_path=None):
        """
        Memulai thread lain untuk menjalankan multiple_ant_colony_system, menggunakan thread utama untuk menggambar
        :return:
        """
        path_queue_for_figure = MPQueue()
        multiple_ant_colony_system_thread = Process(target=self._multiple_ant_colony_system, args=(path_queue_for_figure, max_iterations, file_to_write_path))
        multiple_ant_colony_system_thread.start()

        # Apakah akan menampilkan gambar
        if self.whether_or_not_to_show_figure:
            figure = VrptwAcoFigure(self.graph.nodes, path_queue_for_figure)
            figure.run()
        multiple_ant_colony_system_thread.join()

    def _multiple_ant_colony_system(self, path_queue_for_figure: MPQueue, max_iterations: int, file_to_write_path=None):
        """
        Memanggil acs_time dan acs_vehicle untuk menjelajahi path
        :param path_queue_for_figure:
        :param max_iterations: Jumlah iterasi maksimal
        :param file_to_write_path:
        :return:
        """
        if file_to_write_path is not None:
            file_to_write = open(file_to_write_path, 'w')
        else:
            file_to_write = None

        start_time_total = time.time()

        # Di sini perlu dua antrian, global_path_to_acs_time dan global_path_to_acs_vehicle, digunakan untuk memberitahu acs_time dan acs_vehicle thread tentang best path saat ini, atau untuk menghentikan perhitungan
        global_path_to_acs_time = Queue()
        global_path_to_acs_vehicle = Queue()

        # Antrian lainnya, path_found_queue adalah untuk menerima path yang ditemukan oleh acs_time dan acs_vehicle yang lebih baik dari best path
        path_found_queue = Queue()

        # Menggunakan algoritma nearest neighbor untuk menginisialisasi
        self.best_path, self.best_path_distance, self.best_vehicle_num = self.graph.nearest_neighbor_heuristic()
        path_queue_for_figure.put(PathMessage(self.best_path, self.best_path_distance))

        for iteration in range(max_iterations):
            print('[multiple_ant_colony_system]: iterasi baru')
            start_time_found_improved_solution = time.time()

            # Info best path saat ini, ditempatkan di antrian untuk memberitahu acs_time dan acs_vehicle tentang best path saat ini
            global_path_to_acs_vehicle.put(PathMessage(self.best_path, self.best_path_distance))
            global_path_to_acs_time.put(PathMessage(self.best_path, self.best_path_distance))

            stop_event = Event()

            # acs_vehicle, mencoba dengan self.best_vehicle_num-1 kendaraan untuk menjelajahi, mengunjungi lebih banyak node
            graph_for_acs_vehicle = self.graph.copy(self.graph.init_pheromone_val)
            acs_vehicle_thread = Thread(target=MultipleAntColonySystem.acs_vehicle,
                                        args=(graph_for_acs_vehicle, self.best_vehicle_num-1, self.ants_num, self.q0,
                                              self.beta, global_path_to_acs_vehicle, path_found_queue, stop_event, max_iterations))

            # acs_time mencoba dengan self.best_vehicle_num kendaraan untuk menjelajahi, mencari jarak tempuh yang lebih pendek
            graph_for_acs_time = self.graph.copy(self.graph.init_pheromone_val)
            acs_time_thread = Thread(target=MultipleAntColonySystem.acs_time,
                                     args=(graph_for_acs_time, self.best_vehicle_num, self.ants_num, self.q0, self.beta,
                                           global_path_to_acs_time, path_found_queue, stop_event, max_iterations))

            # Memulai acs_vehicle_thread dan acs_time_thread, ketika mereka menemukan path feasible yang lebih baik, akan mengirimkan ke macs
            print('[macs]: memulai acs_vehicle dan acs_time')
            acs_vehicle_thread.start()
            acs_time_thread.start()

            best_vehicle_num = self.best_vehicle_num

            while acs_vehicle_thread.is_alive() and acs_time_thread.is_alive():

                if path_found_queue.empty():
                    continue

                path_info = path_found_queue.get()
                print('[macs]: menerima info jalur yang ditemukan')
                found_path, found_path_distance, found_path_used_vehicle_num = path_info.get_path_info()
                while not path_found_queue.empty():
                    path, distance, vehicle_num = path_found_queue.get().get_path_info()

                    if distance < found_path_distance:
                        found_path, found_path_distance, found_path_used_vehicle_num = path, distance, vehicle_num

                    if vehicle_num < found_path_used_vehicle_num:
                        found_path, found_path_distance, found_path_used_vehicle_num = path, distance, vehicle_num

                # Jika path yang ditemukan (yang feasible) memiliki jarak yang lebih pendek, perbarui info best path saat ini
                if found_path_distance < self.best_path_distance:

                    # Menemukan solusi yang lebih baik, perbarui start_time
                    start_time_found_improved_solution = time.time()

                    self.print_and_write_in_file(file_to_write, '*' * 50)
                    self.print_and_write_in_file(file_to_write, '[macs]: jarak jalur yang ditemukan (%f) lebih baik daripada best path (%f)' % (found_path_distance, self.best_path_distance))
                    self.print_and_write_in_file(file_to_write, 'waktu yang dibutuhkan %0.3f detik sejak multiple_ant_colony_system berjalan' % (time.time()-start_time_total))
                    self.print_and_write_in_file(file_to_write, '*' * 50)
                    if file_to_write is not None:
                        file_to_write.flush()

                    self.best_path = found_path
                    self.best_vehicle_num = found_path_used_vehicle_num
                    self.best_path_distance = found_path_distance

                    # Jika perlu menggambar gambar, kirim best path yang ditemukan ke program gambar
                    if self.whether_or_not_to_show_figure:
                        path_queue_for_figure.put(PathMessage(self.best_path, self.best_path_distance))

                    # Memberitahu acs_vehicle dan acs_time thread tentang best path dan best path distance yang ditemukan
                    global_path_to_acs_vehicle.put(PathMessage(self.best_path, self.best_path_distance))
                    global_path_to_acs_time.put(PathMessage(self.best_path, self.best_path_distance))

                # Jika, kedua thread ini menggunakan kendaraan lebih sedikit, hentikan kedua thread ini, mulai iterasi berikutnya
                # Kirim info stop ke acs_time dan acs_vehicle thread
                if found_path_used_vehicle_num < best_vehicle_num:

                    # Menemukan solusi yang lebih baik, perbarui start_time
                    start_time_found_improved_solution = time.time()
                    self.print_and_write_in_file(file_to_write, '*' * 50)
                    self.print_and_write_in_file(file_to_write, '[macs]: jumlah kendaraan jalur yang ditemukan (%d) lebih baik daripada best path (%d), jarak jalur yang ditemukan adalah %f'
                          % (found_path_used_vehicle_num, best_vehicle_num, found_path_distance))
                    self.print_and_write_in_file(file_to_write, 'waktu yang dibutuhkan %0.3f detik sejak multiple_ant_colony_system berjalan' % (time.time() - start_time_total))
                    self.print_and_write_in_file(file_to_write, '*' * 50)
                    if file_to_write is not None:
                        file_to_write.flush()

                    self.best_path = found_path
                    self.best_vehicle_num = found_path_used_vehicle_num
                    self.best_path_distance = found_path_distance

                    if self.whether_or_not_to_show_figure:
                        path_queue_for_figure.put(PathMessage(self.best_path, self.best_path_distance))

                    # Hentikan acs_time dan acs_vehicle thread
                    print('[macs]: mengirim info stop ke acs_time dan acs_vehicle')
                    # Memberitahu acs_vehicle dan acs_time thread tentang best path dan best path distance yang ditemukan
                    stop_event.set()

        print('[run_multiple_ant_colony_system]: selesai')

    @staticmethod
    def print_and_write_in_file(file_to_write=None, message='pesan default'):
        if file_to_write is None:
            print(message)
        else:
            print(message)
            file_to_write.write(str(message)+'\n')
