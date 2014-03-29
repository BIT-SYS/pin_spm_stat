#!/usr/bin/python
#-*- encoding:utf-8 -*-

# TODO:
# 多文件读取
# 同一张图每条线的不同color
# 每张图的x,y轴及caption

import sys
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np    
from pylab import *
'''
gStats结构由多个dict组成
gStats[strategy][SPM_size][Block_size][prob][count][hit_cnt/hit_ratio/total_cnt/replace_cnt/replace_ratio]
gStats["All"]["64"]["1"]["0.01"]["2"]["hit_cnt"]
不论是哪一种采样方法都有这几项
chartdemo:
http://matplotlib.org/examples/mplot3d/polys3d_demo.html #多列2维图
http://matplotlib.org/examples/mplot3d/wire3d_animation_demo.html #动画，不需要
http://matplotlib.org/examples/mplot3d/trisurf3d_demo.html #1.2才有
http://matplotlib.org/examples/mplot3d/wire3d_demo.html
'''
def genFloatTuple(start,stop,step):
    res = []
    f = start
    while f <= stop:
        res.append(f)
        f += step
    return tuple(res)

###############################################

All_method = "All_Alt"
#All_method = "All"

gStats = {}
sizeTuple = (64,128,256,512)
blksizeTuple = (1,2,4)
countTuple = (2,4,8,16)
countStep = 2
freqTuple = (0.005,0.03)
freqRandomTuple = genFloatTuple(freqTuple[0],freqTuple[1],0.002)
freqBothTuple = genFloatTuple(0.01,0.1,0.005)
#单一方法的“无效”的参数值
freqDefault = "0"
countDefault = "0"
###############################################
fig_cnt_vs_both = plt.figure() #第一张图片
plt.xlabel("count")
plt.ylabel("spm size")
fig_prob_vs_both = plt.figure() # 又一张图片

ax_cvb = fig_cnt_vs_both.add_subplot(111, projection='3d')
ax_pvb = fig_prob_vs_both.add_subplot(111, projection='3d')
#ax_pvb.set_color_cycle(['c', 'm', 'y', 'k'])
#-----------colors-------------

def getColorlist(count):

    NUM_COLORS = count
    #http://matplotlib.org/examples/pylab_examples/show_colormaps.html
    cm = get_cmap('gist_rainbow')
    #cm = get_cmap('Accent')
    colors = []
    for i in range(NUM_COLORS):
        color0 = cm(1.*i/NUM_COLORS)  # color will now be an RGBA tuple
        colors.append(color0)
        #print color0
    return colors    


def drawProbVsBoth(stat="hit_ratio"):
    clist = getColorlist(len(countTuple) + 1)
    print "clist:",clist
    i = 0
    legends = []
    # 综合方法
    # 取得x,y的tuple
    for cnt in countTuple:
        #取出z的ndarray,需要x,y
        Z_list = []
        for s in sizeTuple:
            tmpRow = []
            for freq in genFloatTuple(0.01,0.1,0.005):
                tmpRow.append(gStats[All_method][str(s)]["2"][str(freq)][str(cnt)][stat])
            Z_list.append(tmpRow)
        Z_array = np.array(Z_list,np.float)
        #print Z_array
        print "Z_array shape:",Z_array.shape
        print "clist: ",clist[i]
        draw(ax_pvb,freqBothTuple,sizeTuple,Z_array,color=clist[i])
        legends.append("R&C: count size:"+str(cnt))
        i += 1
    # ==== 随机方法: spmsize x freq ====
    Z_list = []
    for s in sizeTuple:
        tmpRow = []
        for freq in freqRandomTuple:
            tmpRow.append(gStats["Prob"][str(s)]["2"][str(freq)][countDefault][stat])
        Z_list.append(tmpRow)
    Z_array = np.array(Z_list, np.float)
    #draw(ax_pvb, freqRandomTuple, sizeTuple, Z_array, color=clist[i])
    draw(ax_pvb, freqRandomTuple, sizeTuple, Z_array, color='black')
    legends.append("random")
    ax_pvb.legend(tuple(legends), loc='upper left')
    #print legends
    #fig_prob_vs_both.legend()
def drawCountVsBoth(stat="hit_ratio"):
    clist = getColorlist(len(freqBothTuple) + 1)
    i = 0
    legends = []
    # # ==== 计数方法：spmsize x count ====
    Z_list = []
    for s in sizeTuple:
        tmpRow = []
        for cnt in countTuple:
            tmpRow.append(gStats["Count"][str(s)]["2"][freqDefault][str(cnt)][stat])
        Z_list.append(tmpRow)
    Z_array = np.array(Z_list, np.float)
    #draw(ax_cvb, countTuple, sizeTuple, Z_array, color=clist[i])
    draw(ax_cvb, countTuple, sizeTuple, Z_array, color='black')
    legends.append("count method")
    # #==== 综合方法: spmsize x count ====
    # for freq in genFloatTuple(0.01,0.1,0.01):
    for freq in freqBothTuple:
        i += 1
        # 只输出一半数量的统计图,防止重叠太多
        if i % 2 == 0:
            continue
        Z_list = []
        for s in sizeTuple:
            tmpRow = []
            for cnt in countTuple:
                tmpRow.append(gStats[All_method][str(s)]["2"][str(freq)][str(cnt)][stat])
            Z_list.append(tmpRow)
        Z_array = np.array(Z_list, np.float)
        draw(ax_cvb, countTuple, sizeTuple, Z_array,color=clist[i])
        legends.append("R&C: freq="+str(freq))
    ax_cvb.legend(tuple(legends),loc='upper left')
##############################################
def do_main(filename,progname):
    ax_pvb.set_title('[ ' + progname + " ] R vs R&C",fontsize=30, color='black')
    ax_pvb.set_xlabel('sampling frequence')
    ax_pvb.set_ylabel('total size')
    ax_cvb.set_title('[ ' + progname + " ] C vs R&C",fontsize=30, color='black')
    ax_cvb.set_xlabel('count threshold')
    ax_cvb.set_ylabel('total size')
    do_file_analyse(filename)
    drawProbVsBoth("hit_ratio")
    drawCountVsBoth("hit_ratio")
    figshow()
            
def getZndarray(stategy,ta,tb,statItem):
    for row in ta:
        tmpRow = []
        for col in tb:
            getResult(strategy,)

def getResult(strategy,spmsize,blksize,prob,count,result):
    if strategy in gStats:
        if str(spmsize) in gStats[strategy]:
            if str(blksize) in gStats[strategy][spmsize]:
                if str(prob) in gStat[strategy][spmsize][blksize]:
                    if str(count) in gStat[strategy][str(spmsize)][str(blksize)][str(prob)]:
                        if result in gStats[strategy][str(spmsize)][str(blksize)][str(prob)][str(count)]:
                            return gStats[strategy][str(spmsize)][str(blksize)][str(prob)][str(count)][result]
    else:
        print "result not found"
    return 0.0

def floatRange(start,stop,step):
    r = start
    while r <= stop:
        yield r
        r += step

    
def do_file_analyse(fname):
    '''
    read a single file
    file format:
    每次实验隔一个空行
    
    strategy All
    SPM_size 64
    Block_size 1
    Probability 0.01
    Count_threshold 2
    hit_ratio 0.949639
    replace_ratio 0.000248508
    total_cnt 148123929
    hit_cnt 140664269
    replace_cnt 36810
    '''

    dictTmp = {}
    linum = 1
    with open(fname) as fobj:
        for line in fobj:
            if line != '\n':
                ret = line.split()
                if len(ret) < 2:
                    print "linum:",linum
                    print "line:",line
                    print "line split error, len=",len(ret)
                    exit(-1)
                #print "ret:"
                #print ret
                dictTmp[ret[0]] = ret[1]
                pass
            else:#dictTmp包含了所有读取的数据，由名称作为idx
                if len(dictTmp) != 0:
                    strStrategy = dictTmp["strategy"]
                    if strStrategy not in gStats:
                        gStats[strStrategy] = {}
                    if strStrategy == "Count":
                        defaultProbVar = True
                        defaultCntVar = False
                    elif strStrategy == "Prob":
                        defaultProbVar = False
                        defaultCntVar = True
                    else:
                        defaultProbVar = False
                        defaultCntVar = False

                    nSPM_size = dictTmp["SPM_size"]
                    if nSPM_size not in gStats[strStrategy]:
                        gStats[strStrategy][nSPM_size] = {}
                    nBlock_size = dictTmp["Block_size"]
                    if nBlock_size not in gStats[strStrategy][nSPM_size]:
                        gStats[strStrategy][nSPM_size][nBlock_size] = {}
                    if defaultProbVar:
                        strProb = "0"
                    else:
                        strProb = dictTmp["Probability"]
                    if strProb not in gStats[strStrategy][nSPM_size][nBlock_size]:
                        gStats[strStrategy][nSPM_size][nBlock_size][strProb] = {}
                    if defaultCntVar:
                        strCount = "0"
                    else:
                        strCount = dictTmp["Count_threshold"]
                    if strCount not in gStats[strStrategy][nSPM_size][nBlock_size][strProb]:
                        gStats[strStrategy][nSPM_size][nBlock_size][strProb][strCount] = {}
                    gStats[strStrategy][nSPM_size][nBlock_size][strProb][strCount] = dict(dictTmp)
            linum += 1
        pass

def draw(ax,x_tuple, y_tuple, z_ndarray,color=None):
    drawWire3d(ax,x_tuple,y_tuple,z_ndarray,1,1,color=color)

def drawWire3d(ax, x_tuple,y_tuple,z_ndarray,row_step,col_step,color=None):
#http://matplotlib.org/api/pyplot_api.html
#没有解决多个图像间的间距，其他都ok
    # 参数必须是numpy.ndarray,而不能是tuple
    xndarry,yndarry = getNdarry(x_tuple,y_tuple)
    #print "x shape:",xndarry.shape
    #print "color: ",color
    if color == None:
        ax.plot_wireframe(xndarry,yndarry,z_ndarray,rstride=row_step,cstride=col_step,linewidth='10.0')
    else:
        ax.plot_wireframe(xndarry,yndarry,z_ndarray,rstride=row_step,cstride=col_step,color=color,label="fun")
    #plt.show()

def figshow():
    plt.show()
#
def getNdarry(x_tuple,y_tuple):
    x_list = list(x_tuple)
    #print x_list
    y_list = list(y_tuple)
    x_ndarray = []
    nx = len(x_list)
    ny = len(y_list)
    resx = np.arange(nx * ny)
    resx.dtype=float
    resy = np.arange(nx * ny)
    resx = resx.reshape((ny,nx))
    resy = resy.reshape((ny,nx))
    resy = np.ones_like(resy)
    for i in range(ny):
        resx[i] = x_list
        resy[i] *= y_list[i]
    #print resx
    #print resy
    return resx,resy

################################################################################
if __name__ == "__main__":
    if len(sys.argv) <= 2:
        print 'usage: <output file> <benchmark name>'
        exit(-1)
    if len(sys.argv) > 3 and sys.argv[3] == 'n':
        All_method = "All"
    do_main(sys.argv[1],sys.argv[2])
    
    #print gStats["All"]["64"]["1"]["0.01"]["2"]["hit_cnt"]
