import argparse
import cusine_predictor
import sys
import json

def main(args):
    n=args.N
    l=args.ingredient
    result=cusine_predictor.cuisine_predictor(n,l)
    sys.stdout.write(json.dumps(result, indent=4))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--N',type=int,required=True)
    parser.add_argument('--ingredient',type=str,required=True,action='append')
    args = parser.parse_args()
    main(args)
