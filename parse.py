import tarfile
import re
import numpy
from io import StringIO
import os
import csv
import multiprocessing

from models import Star, ProtoStar
import config

spec = re.compile(r"spec$")
txt = re.compile(r"txt$")
teff = re.compile(r"Teff\s*=\s*'([0-9]+)'")
spaces = re.compile(r" +", re.MULTILINE)
leading_spaces = re.compile(r"^ +", re.MULTILINE)

class EmptyBucketException(RuntimeError):
    pass

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
    return resample(config.startA, config.endA, config.sampleSize, ProtoStar(temp, data))

def parseManyTars(dirName):
    dirContent = os.listdir(dirName)
    pool = multiprocessing.Pool()
    return pool.imap(
        parseTar,
        [os.path.join(dirName,file) for file in dirContent],
        1
    )

def middle(arr):
    if not arr:
        raise EmptyBucketException("Oh no")
    return sum(arr)/len(arr)

def resample(start, end, count, protostar):
    j = 0
    dist = (end - start) / count
    result = numpy.empty(count)
    for i in range(count):
        arr = []
        currend = start + (i + 1) * dist
        while protostar.wavelen[j] < currend:
            arr.append(protostar.spec[j])
            j+=1
        result[i] = middle(arr)
    star = Star(protostar.temp, result)
    return star

def parseStarDir(dirName, targetDirName):
    stars = parseManyTars(dirName)
    for i, star in enumerate(stars):
        filename = '{:05d}.star'.format(i)
        star.save(os.path.join(targetDirName, filename))
        print(star)

def loadStars(dirName):
    fileList = os.listdir(dirName)
    return [Star.load(os.path.join(dirName, file)) for file in fileList]

def main():
    parseStarDir(os.args[1], os.args[2])

if __name__ == '__main__':
    main()
