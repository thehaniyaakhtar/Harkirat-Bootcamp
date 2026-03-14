import java.util.Scanner;

public class Main{
    public static void PrintMessage(int n){
        for(int i = 0; i < n; i++){
            System.out.println("I am learning functions");
        }
    }
    
    public static void main(String[] args){
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        PrintMessage(n);
    }
}
