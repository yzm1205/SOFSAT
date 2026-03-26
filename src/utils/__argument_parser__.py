import argparse


def get_arguments() -> argparse.Namespace:
    """
    Parse all the arguments
    :return:
    """
    # from classical_encoders import MODEL_ENCODER_MAPPING

    # Initialize the parser
    parser = argparse.ArgumentParser(description="Arugments for Sanjeev's paper")

    # Program Arguments
    parser.add_argument(
        "--model",
        type=str,
        help="Model used for computing similarity",
    )

    parser.add_argument("--output_dir", help="Output directory", default="./Results")
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch Size",
    )
    
    parser.add_argument(
        "--gpu", 
        dest="gpu", 
        default="cuda:0", 
        help="GPU to run the model"
    )

    # Parse the arguments
    args = parser.parse_args()
    return args
