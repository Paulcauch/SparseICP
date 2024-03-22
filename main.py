#MAIN 

import argparse

from Tests import *


def main():
    parser = argparse.ArgumentParser(description="Run SICP tests on different datasets.")
    parser.add_argument("--test", choices=["bunny_classical", "bunny_plane", "owls_plane"], help="Specify which test to run.")
    parser.add_argument("--p", type=float, default=1.0, help="Set the 'p' parameter for SICP.")
    parser.add_argument("--ite", type=int, default=50, help="Set the number of iterations for SICP.")
    parser.add_argument("--bunny", choices=["p", "vp", "r"], default="p", help="Specify the bunny dataset variant for classical SICP test.")

    args = parser.parse_args()

    if args.test == "bunny_classical":
        print("Running Classical SICP Test on Bunny Dataset...")
        bunny_classical_SICP(args.p, args.bunny, args.ite)
    elif args.test == "bunny_plane":
        print("Running SICP Plane Test on Bunny Dataset...")
        bunny_SICP_plane(args.p, args.ite)
    elif args.test == "owls_plane":
        print("Running SICP Plane Test on Owls Dataset...")
        owls_SICP_plane(args.p, args.ite)

if __name__ == "__main__":
    main()