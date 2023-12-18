from unittest import TestCase, main
import os
from internal.methods import u_init, u_left


class ConditionTest(TestCase):

    # def test_u_init(self):
    #     self.assertEqual(u_init(0.1, 0.1), 11.09)

    # def test_u_left(self):
    #     self.assertEqual(u_left(0.1), 11)

    def test_4_processes(self):
        self.assertEqual(os.system("mpiexec --oversubscribe -n 4 python ../dev/mpi_app.py"), 0)

    def test_6_processes(self):
        self.assertEqual(os.system("mpiexec --oversubscribe -n 6 python ../dev/mpi_app.py"), 0)

    def test_8_processes(self):
        self.assertEqual(os.system("mpiexec --oversubscribe -n 8 python ../dev/mpi_app.py"), 0)

if __name__=="__main__":
    main()
