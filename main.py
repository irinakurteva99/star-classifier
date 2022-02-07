import tarfile
import re
import numpy
from io import StringIO
import os

spec = re.compile(r"spec$")
txt = re.compile(r"txt$")
teff = re.compile(r"Teff\s*=\s*'([0-9]+)'")
spaces = re.compile(r" +", re.MULTILINE)
leading_spaces = re.compile(r"^ +", re.MULTILINE)

class Star:
    def __init__(self, temp, spec):
        self.temp = temp
        self.spec = spec
    def __repr__(self):
        return "Temp: {}".format(self.temp)

def readFile(tr, file):
    reader = tr.extractfile(file)
    content = reader.read()
    return content

def openTar(fileName):
    tr = tarfile.open(fileName)
    files = tr.getnames()

    return {file: readFile(tr, file).decode("utf-8") for file in files}

def parseTar(tarName):
    filesContent = openTar(tarName)
    specFile = None
    txtFile = None
    for k, v in filesContent.items():
        if spec.search(k):
            specFile = v 
        if txt.search(k):
            txtFile = v
    if not specFile or not txtFile:
        raise RuntimeError("WTF")
    temp = int(teff.search(txtFile).group(1))
    specFile = re.sub(spaces, ' ', specFile)
    specFile = re.sub(leading_spaces, '', specFile)
    data = numpy.genfromtxt(StringIO(specFile), delimiter=" ")
    return Star(temp, data[:,2])

def parseManyTars(dirName):
    dirContent = os.listdir(dirName)
    return (parseTar(os.path.join(dirName,file)) for file in dirContent)

def main(dirName):
    stars = parseManyTars(dirName)
    for star in stars:
        print(star)
