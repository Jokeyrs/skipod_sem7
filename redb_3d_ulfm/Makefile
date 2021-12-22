redb_3d_ulfm: redb_3d_ulfm.out
	mpirun -np 6 --mca shmem posix --mca opal_event_include poll --map-by :OVERSUBSCRIBE --with-ft ulfm redb_3d_ulfm.out 100 100 2
redb_3d_ulfm.out: redb_3d_ulfm.c
	mpicc redb_3d_ulfm.c -o redb_3d_ulfm.out

clean:
	rm -rf redb_3d_ulfm.out