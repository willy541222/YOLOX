#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

from loguru import logger
import sys
sys.path.append(r'D:/YOLOX')
from yolox.exp import get_exp
from yolox.utils import get_model_info
from torchsummary import summary
import argparse


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )

    return parser


def main(exp):

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, (640, 640))))
    # model.eval()
    # print(model)
    summary(model, (3, 640, 640))


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, None)

    main(exp)
