import java.util.*;

public class Main{
    public static void main(String[] args){
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        int[] arr = new int[n];
        
        for(int i = 0; i < n; i++){
            arr[i] = sc.nextInt();
        }
        
        HashMap<Integer, Integer> freq = new HashMap<>();
        
        for(int num : arr){
            freq.put(num, freq.getOrDefault(num, 0) + 1);
        }
        
        boolean found = false;
        
        for(int num : arr){
            if(freq.get(num) == 1){
                System.out.print(num + " ");
                found = true;
            }
        }
        if(!found){
            System.out.println();
        }  
        
    }
}
