from grasp_scorer import score


if __name__ == "__main__":
    grade = 1 if score() > 0.72 else 0
    with open('./grade.txt', 'w') as f:
        f.write('Grade {}\n'.format(grade))
