from pyntcloud import PyntCloud
from pyntcloud.io.ply import write_ply


import glob
import os
import tqdm
import click



def pcd_to_ply(path_in, path_out):
    cloud = PyntCloud.from_file(path_in) # :PyntCloud
    write_ply(path_out, points=cloud.points, as_text=True)


# write_ply('/home/sandro/Dokumente/WHo/ply/test.ply'write_ply(), cloud)

@click.command()
@click.argument('input-folder')
@click.argument('output-folder')
def main(input_folder, output_folder):

    '''
    convert folder to ply
    '''

    files = sorted(glob.glob(os.path.join(input_folder, '*.pcd')))
    for file in tqdm.tqdm(files):
        path_out = os.path.basename(file)
        path_out = path_out.replace('.pcd', '.ply')
        path_out = os.path.join(output_folder, path_out)
        pcd_to_ply(file, path_out)


if __name__ == '__main__':
    main()

