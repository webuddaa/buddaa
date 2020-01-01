import argparse

parser = argparse.ArgumentParser(description='XXXXX')
parser.add_argument('--name', type=str, default='frank')
parser.add_argument('--age', type=int, default=20)
args = parser.parse_args()

print("i love %s, age: %s" % (args.name, args.age))