with open('d.py', 'r') as f_in:
    with open('d1.py', 'w') as f_out:
        for line in f_in:
            if not line.strip().startswith('#'):
                f_out.write(line)
