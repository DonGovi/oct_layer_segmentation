import wx
import os


class octUI(wx.Frame):

    def __init__(self, parent):
        wx.Frame.__init__(self, parent, id=0, title="OCT Layer Segmentation", size=(300, 300))

        panel = wx.Panel(self, id=1)
        sizer = wx.BoxSizer(wx.VERTICAL)
        panel.SetSizer(sizer)
        self.Center()
        
        self.save_button = wx.Button(panel, id=2, label="save", pos=(20, 20))
        self.load_button = wx.Button(panel, id=3, label="load", pos=(120, 20))

        '''
        self.Bind(wx.EVT_BUTTON, self.OnClick, self.save_button)
        self.inputText = wx.TextCtrl(panel, -1, "", pos=(100, 10), size=(150, -1), style=wx.TE_READONLY)
        '''
    
    def OnClick(self, event):
        self.inputText.Value = "Hello World"
    


class LayerSegApp(wx.App):
    def OnInit(self):
        print("Starting Event Loop...")
        self.frame = octUI(None)
        self.frame.Show(True)
        
        return True
    
    def OnExit(self):
        print("Ending Event Loop...")
        import time
        time.sleep(2)
        
        return 0



app = LayerSegApp()

app.MainLoop()
