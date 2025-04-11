.PHONY : debug, release, profile

debug:
	cmake -DCMAKE_BUILD_TYPE=Debug -DPROFILE_BUILD=OFF .
	make
	gdb ./basis-choice -ex run

release:
	cmake -DCMAKE_BUILD_TYPE=Release -DPROFILE_BUILD=OFF .
	make
	./basis-choice

profile:
	cmake -DCMAKE_BUILD_TYPE=Release -DPROFILE_BUILD=ON .
	make
	./flamegraph.sh ~/opt/FlameGraph firefox
