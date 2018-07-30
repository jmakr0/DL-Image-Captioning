import json
from argparse import ArgumentParser


def main(input_file, output_file, filter_list):
    # read json
    print("reading file {}".format(input_file))
    with open(input_file, 'r') as fh_in:
        data = json.load(fh_in)

    # process
    if not data['images']:
        print("json does not contain a key 'images'")
        exit(1)
    if not data['annotations']:
        print("json does not contain a key 'annotations")
        exit(1)

    print("\nBefore processing:")
    print("  # images: {}".format(len(data['images'])))
    print("")

    print("Filtering out following images:\n{}".format(filter_list))
    imgs = [img for img in data['images'] if img['id'] not in filter_list]
    annos = [a for a in data['annotations'] if a['image_id'] not in filter_list]
    data['images'] = imgs
    data['annotations'] = annos

    print("\nAfter processing:")
    print("  # images: {}".format(len(data['images'])))
    print("")

    # write json
    print("writing result to file {}".format(output_file))
    with open(output_file, 'w') as fh_out:
        json.dump(data, fh_out)
    exit(0)


if __name__ == "__main__":
    arg_parse = ArgumentParser(description="Filters out specific annotations and images " +
                                           "from the metadata file based on image IDs.")
    arg_parse.add_argument('--negative_image_ids',
                           default=[],
                           type=int,
                           nargs='*',
                           help='Image IDs to filter out')
    arg_parse.add_argument('input',
                           type=str,
                           help="filepath to the original metadata file")
    arg_parse.add_argument('output',
                           type=str,
                           help="filepath, where the result metadata should be saved to")
    args = arg_parse.parse_args()

    main(args.input, args.output, args.negative_image_ids)
