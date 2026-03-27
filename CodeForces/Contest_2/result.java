
import java.util.Scanner;

public class Main{
    public static void main(String[] args){
        Scanner sc = new Scanner(System.in);
        
        int n = sc.nextInt();
        int[] arr = new int[n];
        
        for(int i = 0; i < n; i++){
            arr[i] = sc.nextInt();
        }
        
        int Pass = 0;
        int Fail = 0;
        int marks = sc.nextInt();
        
        for(int i = 0; i < n; i++){
            if(arr[i] >= marks){
                Pass += 1;
            }
            else{
                Fail += 1;
            }
        }
        
        System.out.println("Pass: " + Pass);
        System.out.println("Fail: " + Fail);
    }
}
