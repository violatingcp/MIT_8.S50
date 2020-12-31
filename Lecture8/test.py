import ROOT as r
from ROOT import std,RooDataHist
import csv
import math
from array import array

fOutput="Output.root"

def sigVsMassPlot(masses,pvalues,labels):
    lC0 = r.TCanvas("A","A",800,600)
    leg = r.TLegend(0.55,0.23,0.86,0.47)
    leg.SetFillColor(0)
    lGraphs=[]
    sigmas=[]
    for i0 in range(len(masses)):
        graph1 = r.TGraph(len(masses[i0]),masses[i0],pvalues[i0])
        graph1.SetMarkerStyle(20)
        graph1.GetXaxis().SetTitle("m_{jj} (GeV)")
        graph1.GetYaxis().SetTitle("p^{0} value")
        graph1.SetTitle("Significance vs Mass")
        graph1.SetLineColor(2+i0)
        graph1.SetMarkerColor(2+i0)
        graph1.SetLineWidth(2+i0)
        r.gPad.SetLogy(True)
        graph1.GetYaxis().SetRangeUser(1e-8,1.0)
        if i0 == 0:
            graph1.Draw("alp")
        else:
            graph1.Draw("lp")
        lGraphs.append(graph1)
        leg.AddEntry(graph1,labels[i0],"lp")
    sigmas=[0.317,0.045,0.0027,0.0000633721,0.0000005742]                                                                                                                                                                                    
    lines=[]
    for i0 in range(5):#len(sigmas)):                                                                                                                                                                                                                sigmas.append(1-norm.cdf(i0+1))
        lLine = r.TLine(masses[0][0],sigmas[i0],masses[0][len(masses[0])-1],sigmas[i0])
        lLine.SetLineStyle(r.kDashed)
        lLine.SetLineWidth(2)
        lLine.Draw()
        lPT = r.TPaveText(3500,sigmas[i0],4000,sigmas[i0]+1.5*sigmas[i0])
        lPT.SetFillStyle(4050)
        lPT.SetFillColor(0)
        lPT.SetBorderSize(0)
        lPT.AddText(str(i0+1)+"#sigma")
        lPT.Draw()
        lines.append(lLine)
        lines.append(lPT)

    for pGraph in lGraphs:
        pGraph.Draw("lp")
    leg.Draw()
    lC0.Update()
    lC0.Draw()
    lC0.SaveAs("pvalue_bb1.png")

# build workspace
def workspace(iOutput,iDatas,iFuncs,iCat="cat0"):
    print('--- workspace')
    lW = r.RooWorkspace("w_"+str(iCat))
    for pData in iDatas:
        print('adding data ',pData,pData.GetName())
        getattr(lW,'import')(pData,r.RooFit.RecycleConflictNodes())
    for pFunc in iFuncs:
        print('adding func ',pFunc,pFunc.GetName())
        getattr(lW,'import')(pFunc,r.RooFit.RecycleConflictNodes())
    if iCat.find("pass_cat0") == -1:
        lW.writeToFile(iOutput,False)
    else:
        lW.writeToFile(iOutput)
    return lW
    
# now lets do it with roofit
def drawFrame(iX,iData,iFuncs):
    lCan   = r.TCanvas("qcd","qcd",800,600)
    leg = r.TLegend(0.55,0.63,0.86,0.87)
    lFrame = iX.frame()
    lFrame.SetTitle("")
    lFrame.GetXaxis().SetTitle("m_{jj} (GeV)")
    lFrame.GetYaxis().SetTitle("Events")
    iData.plotOn(lFrame)
    iColor=51
    iFuncs[0].plotOn(lFrame,r.RooFit.LineColor(r.kGreen+1))
    iFuncs[1].plotOn(lFrame,r.RooFit.LineColor(iColor),r.RooFit.LineStyle(r.kDashed))
    leg.SetFillColor(0)
    lFrame.Draw()
    lTmpData  = r.TH1F("tmpData" ,"tmpData" ,1,0,10); lTmpData .SetMarkerStyle(r.kFullCircle);
    lTmpFunc1 = r.TH1F("tmpFunc1","tmpFunc1",1,0,10); lTmpFunc1.SetLineColor(51); 
    lTmpFunc1.SetLineWidth(2); lTmpFunc1.SetLineStyle(r.kDashed);
    lTmpFunc2 = r.TH1F("tmpFunc2","tmpFunc2",1,0,10); lTmpFunc2.SetLineColor(61);                
    lTmpFunc2.SetLineWidth(2); lTmpFunc2.SetLineStyle(r.kDashed);
    leg.AddEntry(lTmpData,"data","lpe")
    leg.AddEntry(lTmpFunc2,"bkg","lp")
    leg.AddEntry(lTmpFunc1,"loss-sideband","lp")
    leg.Draw()
    lCan.Modified()
    lCan.Update()
    lCan.SaveAs(lCan.GetName()+".png")

lData = r.TH1F("data","data",55,105,160)
lData.Sumw2()
label='out.txt'
with open(label,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=' ')
    for row in plots:
        lData.Fill(float(row[1]),float(row[2]))
        #x.append(float(row[1]))
        #y.append(float(row[2]))
        #add gaussian uncertainties                                                                                                 
        #y_err.append(math.sqrt(float(row[2])))
for i0 in range(lData.GetNbinsX()+1):
    lData.SetBinError(i0,math.sqrt(lData.GetBinContent(i0)))
        
#lCan = r.TCanvas("A","A",800,600)        
#lCan.Draw()
#lData.Draw()
#lCan.Update()
lNBins=lData.GetNbinsX()
lX = r.RooRealVar("x","x",100,180)
lX.setBins(lNBins)
lNTot   = r.RooRealVar("bkgnorm","bkgnorm",lData.Integral(),0,3*lData.Integral())
lA0     = r.RooRealVar   ("a0","a0",0.00,-1.,1.)                                                                 
lA1     = r.RooRealVar   ("a1","a1",0.0,-1,1.)                                                                  
lA2     = r.RooRealVar   ("a2","a2",0.0,-1,1)                                                                   
lA3     = r.RooRealVar   ("a3","a3",0.0,-1,1)                                                                   
lA4     = r.RooRealVar   ("a4","a4",0.0,-1,1)                                                                   
lA5     = r.RooRealVar   ("a5","a5",0.0,-1,1)                                                                   
lPoly   = r.RooBernstein("bkg","bkg",lX,r.RooArgList(lA0,lA1,lA2,lA3,lA4,lA5))
#lPoly   = r.RooPolynomial("bkg","bkg",lX,r.RooArgList(lA0,lA1,lA2,lA3,lA4,lA5))                              
lBkg    = r.RooExtendPdf("ebkg", "ebkg",lPoly,lNTot)
lMass   = r.RooRealVar("mass","mass"  ,125,100,180); #lMass.setConstant(r.kTRUE)                                                 
lSigma  = r.RooRealVar("sigma","Width of Gaussian",1.7,0,10); lSigma.setConstant(r.kTRUE)
lGaus   = r.RooGaussian("gauss","gauss(x,mean,sigma)",lX,lMass,lSigma)
lNSig   = r.RooRealVar("signorm","signorm",0.1*lData.Integral(),0,0.3*lData.Integral())
lSig    = r.RooExtendPdf("sig", "sig",lGaus,lNSig)
lSum    = r.RooAddPdf("model", "model", r.RooArgList(lSig, lBkg))
lHData  = r.RooDataHist("data_obs","data_obs", r.RooArgList(lX),lData)

lSum.fitTo(lHData)
drawFrame(lX,lHData,[lBkg,lSum])

#lA0.setConstant(r.kTRUE)
#lA1.setConstant(r.kTRUE)
#lA2.setConstant(r.kTRUE)
#lA3.setConstant(r.kTRUE)
#lA4.setConstant(r.kTRUE)
#lA5.setConstant(r.kTRUE)
lMass.setConstant(r.kTRUE)     

lW = workspace(fOutput,[lHData],[lSum,lBkg])
lW.defineSet("poi","signorm")
bmodel = r.RooStats.ModelConfig("b_model",lW)
bmodel.SetPdf(lW.pdf("model"))
bmodel.SetNuisanceParameters(r.RooArgSet(lA1,lA2,lA3,lA4,lA5,lNTot,lSigma,lMass,lNSig))
bmodel.SetObservables(r.RooArgSet(lX))
bmodel.SetParametersOfInterest(lW.set("poi"))
lW.var("signorm").setVal(0)
bmodel.SetSnapshot(lW.set("poi"))
    
sbmodel = r.RooStats.ModelConfig("s_model",lW)
sbmodel.SetPdf(lW.pdf("model"))
sbmodel.SetNuisanceParameters(r.RooArgSet(lA1,lA2,lA3,lA4,lA5,lNTot,lSigma,lMass,lNSig))
sbmodel.SetObservables(r.RooArgSet(lX))
sbmodel.SetParametersOfInterest(lW.set("poi"))
lW.var("signorm").setVal(lNSig.getVal())
sbmodel.SetSnapshot(lW.set("poi"))

iMin=110
iMax=145
iStep=100
masses =  array( 'd' )
pvalues = array( 'd' )
stepsize = float(iMax-iMin)/float(iStep)
masslist = [iMin + i*stepsize for i in range(iStep+1)]
r.RooStats.AsymptoticCalculator(lHData, sbmodel, bmodel)
for mass in masslist:
    lW.var("mass").setVal(mass)
    ac = r.RooStats.AsymptoticCalculator(lHData, sbmodel, bmodel)
    ac.SetOneSidedDiscovery(True)
    ac.SetPrintLevel(2)
    asResult = ac.GetHypoTest()
    pvalue=asResult.NullPValue()
    if pvalue > 1e-8:
        masses.append(mass)
        pvalues.append(pvalue)
        print(mass,pvalue)

labels=["test"]
print(masses)
sigVsMassPlot([masses],[pvalues],labels)


