import java.io.*;

public class Main{
    public static void main(String[] args) throws Exception{
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        
        char c = br.readLine().charAt(0);
        
        if(c >= 'a' && c <= 'z') System.out.println("Lowercase");
        else if(c >= 'A' && c <= 'Z') System.out.println("Uppercase");
        else if(c >= '0' && c <= '9') System.out.println("Digit");
        else System.out.println("Special");
    }
}
