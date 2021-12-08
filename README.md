# taos
 taos related work


## Conda data processing environment

See [equinox doc](https://github.com/apatlpo/mit_equinox/blob/master/doc/conda.md)

## Ichtyop install

### datarmor

Install java (`jdk-11.0.12_linux-x64_bin.tar.gz`) or via conda:
```
conda create -n ichthy -c conda-forge python=3.8 maven libnetcdf
conda activate ichthy
```

Build [Ichthyop](http://www.ichthyop.org/documentation/):

```
conda activate ichthy
git clone https://github.com/ichthyop/ichthyop.git
cd ichthyop

module unload NETCDF/4.3.3.1-mpt-intel2016
set path = ($path $home/.miniconda3/envs/ichthy/lib)
setenv JAVA_HOME $CONDA_PREFIX
setenv JAVA_LD_LIBRARY_PATH $JAVA_HOME/lib/server

mvn clean install -B package
```


## Run Ichtyop

```
#java -jar target/ichthyop-3.3.10-jar-with-dependencies.jar # to get the console
java -jar target/ichthyop-3.3.10-jar-with-dependencies.jar taos_mars3d.xml
```
