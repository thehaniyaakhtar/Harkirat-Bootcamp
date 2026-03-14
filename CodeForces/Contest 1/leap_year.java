import java.util.Scanner;

public class Main{
    public static void main(String[] args){
        Scanner sc = new Scanner(System.in);
        int a = sc.nextInt();
        
        if(a%400==0 || (a%4==0 && a%100!=00)){
            System.out.print("Yes");
        }
        else{
            System.out.print("No");
        }
    }
}
