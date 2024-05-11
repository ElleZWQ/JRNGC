class TGCModel(object):
    def exper_loss(self, x):
        """
        
        """
        raise NotImplementedError()

    def struct_loss(self):
        """
        
        """
        raise NotImplementedError()
    
    def gate_loss(self):
        """
      
        """
        raise NotImplementedError()
    
    @staticmethod
    def from_train(x, x_eval, **kwargs):
        """
        
        """
        raise NotImplementedError()
    
    def get_gc_metrics(*args, **kwargs):
        """
       
        """
        raise NotImplementedError()