#! /usr/bin/python3

# import time

import struct
import numpy as np
from array import array
from pathlib import Path
from time import perf_counter

from OpenGL import GL

import moderngl
import moderngl_window


# SIZE = 2**27 # 134 217 728
SIZE = 2**20 # 1 048 576
# SIZE = 2**16 # 65 536

class Algorithm:
    LocalBitonicMergeSortExample = 0
    LocalDisperse = 1
    BigFlip = 2
    BigDisperse = 3

class MyWindow(moderngl_window.WindowConfig):
    gl_version = (4, 5)
    samples = 0  # Headless is not always happy with multisampling
    resource_dir = (Path(__file__) / "../../shaders").resolve()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.program = self.load_compute_shader(path='merge_sort_comp.glsl')

        self.arr = np.random.randint(1_000_000, size=SIZE)
        self.buffer = self.ctx.buffer(data=array('I', self.arr))

        self.query = self.ctx.query(samples=False, time=True)

        # self.queries = []

    def sort(self, max_workgroup_size=1024, n=SIZE):
        workgroup_size_x = 1

        if n < max_workgroup_size * 2:
            workgroup_size_x = n // 2;
        else:
            workgroup_size_x = max_workgroup_size;

        h = workgroup_size_x * 2
        assert (h <= n)
        assert (h % 2 == 0)

        workgroup_count = n // ( workgroup_size_x * 2 );

        self.program['u_h'] = h
        self.program['u_algorithm'] = Algorithm.LocalBitonicMergeSortExample
        # with self.query:
        self.program.run(group_x=workgroup_count)
        # self.queries.append(self.query.elapsed * 10e-7)
        # print("query time LocalBitonicMergeSortExample: {:.3f}ms".format(self.query.elapsed * 10e-7))
        h *= 2

        while (h <= n):
            GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT); ## way better than ctx.finish()

            self.program['u_h'] = h
            self.program['u_algorithm'] = Algorithm.BigFlip
            # with self.query:
            self.program.run(group_x=workgroup_count)
            # self.queries.append(self.query.elapsed * 10e-7)
            # print("query time BigFlip: {:.3f}ms".format(self.query.elapsed * 10e-7))

            hh = h // 2
            while hh > 1:
                GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT);

                if hh <= workgroup_size_x * 2:
                    self.program['u_h'] = hh
                    self.program['u_algorithm'] = Algorithm.LocalDisperse
                    # with self.query:
                    self.program.run(group_x=workgroup_count)
                    # self.queries.append(self.query.elapsed * 10e-7)
                    # print("query time LocalDisperse: {:.3f}ms".format(self.query.elapsed * 10e-7))
                    break
                else:
                    self.program['u_h'] = hh
                    self.program['u_algorithm'] = Algorithm.BigDisperse
                    # with self.query:
                    self.program.run(group_x=workgroup_count)
                    # self.queries.append(self.query.elapsed * 10e-7)
                    # print("query time BigDisperse: {:.3f}ms".format(self.query.elapsed * 10e-7))

                hh //= 2

            h *= 2

    def render(self, time, frametime):
        self.buffer.bind_to_storage_buffer(0)

        t1_start = perf_counter()
        self.sort()
        t2_start = perf_counter()

        print()

        data = self.buffer.read_chunks(chunk_size=4, start=0, step=4, count=SIZE)
        data = struct.iter_unpack('I', data)
        data = [v[0] for v in data]
        # print(data)

        is_sorted = np.all(np.diff(data) >= 0)
        print("sorted: {}".format(is_sorted))

        t = (t2_start - t1_start) * 1000
        print(f"GPU Took {t:.3f}ms to sort {SIZE} elements")
        # print("Sum of queries time: {:.3}ms".format(sum(self.queries)))

        print()

        ## pyhon/numpy sort
        t1_start = perf_counter()
        self.arr.sort()
        t2_start = perf_counter()
        is_sorted = np.all(np.diff(self.arr) >= 0)
        print("sorted: {}".format(is_sorted))
        t = (t2_start - t1_start) * 1000
        print(f"CPU Took {t:.3f}ms to sort {SIZE} elements\n")


        self.wnd.close()


if __name__ == "__main__":
    moderngl_window.run_window_config(MyWindow, args=('--window', 'headless'))
