all:
	cd build&&cmake ..&&make&&cd ..

cdt:
	cd build&&cmake -G"Eclipse CDT4 - Unix Makefiles" ..&&cp .project ..&&cp .cproject ..&&cd ..

clean:
	cd build&&cmake ..&&make clean&&cd ..

dist-clean:
	cd build&&rm -r *&&cmake ..&&make clean&&cd ..
