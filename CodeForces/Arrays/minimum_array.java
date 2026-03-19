import java.util.Scanner;

public class Main{
    public static void main(String[] args){
        
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        int[] arr = new int[n];
        
        for(int i = 0; i < n; i++){
            arr[i] = sc.nextInt();
        }
        
        int min = arr[0];
        int pos = 1;
        
        for(int i = 0; i < n; i++){
            if(arr[i] < min){
                min = arr[i];
                pos = i+1;
            }
        }
        System.out.print(min + " " + pos);
    }
}
