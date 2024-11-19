from exercise2_coordinate_descent import *

def main():
    #2a
    x_0=(2,3,4)
    print("2a")
    print("argmin_x1("+str(x_0)+")="+str(argmin_x1(x_0)))
    print("argmin_x2("+str(x_0)+")="+str(argmin_x2(x_0)))
    print("argmin_x3("+str(x_0)+")="+str(argmin_x3(x_0)))
    #2b
    print("2b")
    solutions_1_iteration=coordinate_descent(argmin=[argmin_x1,argmin_x2,argmin_x3],x_0=(1,20,5),max_iter=1)
    print("\nsolution after 1 iteration:")
    print("argmin[0]("+str(x_0)+")="+str(solutions_1_iteration[0]))
    print("argmin[0]("+str(x_0)+")="+str(solutions_1_iteration[1]))
    print("argmin[0]("+str(x_0)+")="+str(solutions_1_iteration[2]))
    solution_100_iteration=coordinate_descent(argmin=[argmin_x1,argmin_x2,argmin_x3],x_0=(1,20,5),max_iter=100)
    print("\nsolution after 100 iterations: "+str(tuple("x"+str(i+1)+"="+str(x) for (i,x) in enumerate(solution_100_iteration))))

if __name__ == "__main__":
    main()