import java.util.Scanner;

public class Main{
    public static void main(String[] args){
        
        Scanner sc = new Scanner(System.in);
        
        int a = sc.nextInt();
        
        if(a > 90){
            System.out.println("Excellent");
        }
        else if(a > 80 && a <= 90){
            System.out.println("Good");
        }
        else if(a > 70 && a <= 80){
            System.out.println("Fair");
        }
        else if(a > 60 && a <= 70){
            System.out.println("Meets Expectations");
        }
        else{
            System.out.println("Below Par");
        }
    }
}



