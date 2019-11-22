prereq:
	# Install dependencies - OSMesa, mmatl's fork of PyOpenGL
	sudo apt update
	wget https://github.com/mmatl/travis_debs/raw/master/xenial/mesa_18.3.3-0.deb
  	sudo dpkg -i ./mesa_18.3.3-0.deb || true
	sudo apt install -f
	git clone https://github.com/mmatl/pyopengl.git
	pip install ./pyopengl

	# Install pyrender
	pip install pyrender

	wget --no-check-certificate https://lmb.informatik.uni-freiburg.de/resources/datasets/ORION/modelnet40_manually_aligned.tar
	mkdir './modelnet'
	tar -xf './modelnet40_manually_aligned.tar' -C '/modelnet'
	rm 'modelnet40_manually_aligned.tar'
	
