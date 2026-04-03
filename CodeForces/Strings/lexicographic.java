import java.io.*;
import java.util.*;
 
public class Main{
    public static void main(String[] args) throws Exception{
        
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        
        String first = br.readLine();
        String second = br.readLine();
        
        if(first.compareTo(second) < 0){
            System.out.println("A");
        }
        else if(first.equals(second)){
            System.out.println("Equal");
        }
        else{
            System.out.println("B");
        }
    }
}
