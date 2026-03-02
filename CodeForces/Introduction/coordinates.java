import java.util.Scanner;
 
public class Main{
    public static void main(String[] args){
        
        Scanner sc = new Scanner(System.in);
        
        int a = sc.nextInt();
        int b = sc.nextInt();
        
        if(a == 0 && b == 0){
            System.out.println("Origin");
        }
        if(a == 0 && b != 0){
            System.out.println("Y axis");
        }
        if(a != 0 && b == 0){
            System.out.println("X axis");
        }
        if(a > 0 && b > 0){
            System.out.println("1st Quadrant");
        }
        if(a < 0 && b > 0){
            System.out.println("2nd Quadrant");
        }
        if(a < 0 && b < 0){
            System.out.println("3rd Quadrant");
        }
        if(a > 0 && b < 0){
            System.out.println("4th Quadrant");
        }
    }
}
 
 
