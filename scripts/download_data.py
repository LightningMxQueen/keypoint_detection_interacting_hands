import wget

base_url = 'https://fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15.s3.amazonaws.com/InterHand2.6M/InterHand2.6M.images.5.fps.v1.0/'
output_path = '../data/'

wget.download( base_url + 'InterHand2.6M.images.5.fps.v1.0.tar.CHECKSUM', out=output_path)
wget.download( base_url + 'unzip.sh', out=output_path)
wget.download( base_url + 'verify_download.py', out=output_path)

for part1 in ('a', 'b'):
    for part2 in ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'):
        if part1 == 'b' and part2 == 's':
            break
        wget.download(base_url + 'InterHand2.6M.images.5.fps.v1.0.tar.part' + part1 + part2, out= output_path)
