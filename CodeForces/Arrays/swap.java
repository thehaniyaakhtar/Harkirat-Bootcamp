import java.util.Scanner;

public class Main{
    public static void main(String[] args){
        
        Scanner sc = new Scanner(System.in);
        int t = sc.nextInt();
        
        while(t-->0){
            int n = sc.nextInt();
            int[] arr = new int[n];
            
            for(int i = 0; i < n; i++){
                arr[i] = sc.nextInt();
            }
            
            for(int i = 0; i < n-1; i+=2){
                int temp = arr[i];
                arr[i] = arr[i+1];
                arr[i+1] = temp;
            }
            
            for(int i = 0; i < n; i++){
                System.out.print(arr[i] + " ");
            }
            System.out.println();
        }
    }
}
