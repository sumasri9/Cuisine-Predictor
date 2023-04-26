import cusine_predictor
import unittest


class Test(unittest.TestCase):
    def test1(self):
        n=5
        l=['banana','rice krispies','paprica']
        result = cusine_predictor.cuisine_predictor(n,l)
        self.assertIsNotNone(result,"didn't return any value")
    def test2(self):
        n=5
        l=['banana','rice krispies','paprica']
        result = cusine_predictor.cuisine_predictor(n,l)
        if (type(result)==dict):
            true = 1
        else:
            true = 0
        assert(true==1)
    def test3(self):
        n=5
        l=['banana','rice krispies','paprica']
        result = cusine_predictor.cuisine_predictor(n,l)
        if(result.setdefault('cuisine')):
            true=1
        else:
            true=0
        assert(true==1)
    def test4(self):
        n=5
        l=['banana','rice krispies','paprica']
        result = cusine_predictor.cuisine_predictor(n,l)
        if(result.setdefault('score')):
            true=1
        else:
            true=0
        assert(true==1)
    def test5(self):
        n=5
        l=['banana','rice krispies','paprica']
        result = cusine_predictor.cuisine_predictor(n,l)
        if(result.setdefault('closest')):
            true=1
        else:
            true=0
        assert(true==1)
        

if __name__ == '__main__':
    unittest.main()


