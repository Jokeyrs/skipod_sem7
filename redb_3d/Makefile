redb_3d: redb_3d.out
	mpirun -np 6 --mca shmem posix --mca opal_event_include poll --map-by :OVERSUBSCRIBE --with-ft ulfm redb_3d.out 100 100 2
redb_3d.out: redb_3d.c
	mpicc redb_3d.c -o redb_3d.out

clean:
	rm -rf redb_3d.out