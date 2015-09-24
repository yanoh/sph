all : movie

sph_water : sph.cpp water.h
	g++ -DPARAM=\"water.h\" -DNDEBUG -Wall -pedantic -std=c++11 -Ofast -o $@ sph.cpp

sph_oil : sph.cpp oil.h
	g++ -DPARAM=\"oil.h\" -DNDEBUG -Wall -pedantic -std=c++11 -Ofast -o $@ sph.cpp

sph_gel : sph.cpp gel.h
	g++ -DPARAM=\"gel.h\" -DNDEBUG -Wall -pedantic -std=c++11 -Ofast -o $@ sph.cpp

movie_water : sph_water
	sh -c 'time ./sph_water'
	octave plot.m
	ffmpeg -framerate 60  -i 'sph-%04d.png' sph_water-1.mp4
	ffmpeg -framerate 120 -i 'sph-%04d.png' sph_water-2.mp4
	ffmpeg -framerate 240 -i 'sph-%04d.png' sph_water-4.mp4

movie_oil : sph_oil
	sh -c 'time ./sph_oil'
	octave plot.m
	ffmpeg -framerate 60  -i 'sph-%04d.png' sph_oil-1.mp4
	ffmpeg -framerate 120 -i 'sph-%04d.png' sph_oil-2.mp4
	ffmpeg -framerate 240 -i 'sph-%04d.png' sph_oil-4.mp4

movie_gel : sph_gel
	sh -c 'time ./sph_gel'
	octave plot.m
	ffmpeg -framerate 60  -i 'sph-%04d.png' sph_gel-1.mp4
	ffmpeg -framerate 120 -i 'sph-%04d.png' sph_gel-2.mp4
	ffmpeg -framerate 240 -i 'sph-%04d.png' sph_gel-4.mp4

movie : movie_gel movie_oil movie_water 

clean :
	rm -f sph_water sph_oil sph_gel *.mat *.png *.mp4
