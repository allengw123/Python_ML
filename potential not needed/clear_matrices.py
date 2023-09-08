import os

patients = ["pro00087153_0003", "pro00087153_0004", "pro00087153_0005",
 "pro00087153_0042", "pro00087153_0013", "pro00087153_0017", "pro00087153_0018", 
 "pro00087153_0021", "pro00087153_0022", "pro00087153_0030", "pro00087153_0023",
 "pro00087153_0015", "pro00087153_0020", "pro00087153_0024", "pro00087153_0025",
"pro00087153_0026", "pro00087153_0027", "pro00087153_0028", "pro00087153_0029", "pro00087153_0036", "pro00087153_0043"]

output_file = "./output"
for root, dirs, files in os.walk(output_file):
    for name in dirs:
        if (name in patients):
            current_pt = name
            pt_directory = os.path.join(output_file, current_pt)
        else:
            continue
        for root2, dirs2, files2 in os.walk(pt_directory):
            for file2 in files2:
                if ("model_conf_matrix" in file2):
                    file_path = os.path.join(pt_directory, file2)
                    os.system("rm -f %s" % file_path)
